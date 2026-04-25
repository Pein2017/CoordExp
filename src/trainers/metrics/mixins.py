from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.coord_tokens.codec import get_coord_token_ids
from src.data_collators.token_types import TokenType
from src.trainers.monitoring.instability import InstabilityMonitorMixin  # noqa: F401


def _resolve_text_vocab_size(model: Any) -> int | None:
    config = getattr(model, "config", None)
    text_cfg = getattr(config, "text_config", None)
    text_vocab = getattr(text_cfg, "vocab_size", None)
    if isinstance(text_vocab, int) and text_vocab > 0:
        return int(text_vocab)
    vocab = getattr(config, "vocab_size", None)
    if isinstance(vocab, int) and vocab > 0:
        return int(vocab)
    return None


def _resolve_embedding_rows(model: Any) -> int | None:
    get_emb = getattr(model, "get_input_embeddings", None)
    if not callable(get_emb):
        return None
    try:
        embeddings = get_emb()
    except Exception:
        return None
    weight = getattr(embeddings, "weight", None)
    shape = getattr(weight, "shape", None)
    if not shape or len(shape) < 1:
        return None
    try:
        rows = int(shape[0])
    except (TypeError, ValueError):
        return None
    return rows if rows > 0 else None


def _validate_batch_contract(
    *,
    model: Any,
    inputs: Mapping[str, Any],
    template: Any = None,
) -> None:
    input_ids = inputs.get("input_ids")
    flat_input_ids: torch.Tensor | None = None
    batch_size: int | None = None
    seq_len: int | None = None
    if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
        embed_rows = _resolve_embedding_rows(model)
        if embed_rows is not None:
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= embed_rows:
                raise ValueError(
                    "Batch input_ids violate embedding range before model forward: "
                    f"min_id={min_id} max_id={max_id} embed_rows={embed_rows} "
                    f"shape={tuple(input_ids.shape)}"
                )
        flat_input_ids = input_ids.reshape(-1)
        if input_ids.ndim == 1:
            batch_size = 1
            seq_len = int(input_ids.shape[0])
        elif input_ids.ndim >= 2:
            batch_size = int(input_ids.shape[0])
            seq_len = int(input_ids.shape[-1])

    labels = inputs.get("labels")
    if isinstance(labels, torch.Tensor) and labels.numel() > 0:
        vocab_size = _resolve_text_vocab_size(model)
        if vocab_size is not None:
            invalid = (labels < -100) | ((labels != -100) & (labels >= vocab_size))
            if bool(invalid.any().item()):
                bad_values = torch.unique(labels[invalid]).detach().cpu().tolist()
                bad_values = [int(v) for v in bad_values[:16]]
                raise ValueError(
                    "Batch labels violate causal-LM target range before model forward: "
                    f"vocab_size={vocab_size} bad_values={bad_values} "
                    f"shape={tuple(labels.shape)}"
                )

    image_grid = inputs.get("image_grid_thw")
    pixel_values = inputs.get("pixel_values")
    image_token_id = getattr(template, "image_token_id", None)
    if isinstance(image_grid, torch.Tensor) and image_grid.numel() > 0:
        if image_grid.ndim != 2 or int(image_grid.shape[-1]) != 3:
            raise ValueError(
                "Batch image_grid_thw must have shape [num_images, 3] before model forward: "
                f"shape={tuple(image_grid.shape)}"
            )
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim >= 1:
            expected_visual_rows = int(image_grid.prod(dim=-1).sum().item())
            actual_visual_rows = int(pixel_values.shape[0])
            if actual_visual_rows != expected_visual_rows:
                raise ValueError(
                    "Batch pixel_values/image_grid_thw visual-row mismatch before model forward: "
                    f"expected_rows={expected_visual_rows} actual_rows={actual_visual_rows} "
                    f"pixel_values.shape={tuple(pixel_values.shape)} image_grid_thw.shape={tuple(image_grid.shape)}"
                )
        if flat_input_ids is not None and isinstance(image_token_id, int):
            processor = getattr(template, "processor", None)
            image_processor = getattr(processor, "image_processor", None)
            merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
            if merge_size <= 0:
                merge_size = 1
            expected = int((image_grid.prod(dim=-1) // (merge_size**2)).sum().item())
            actual = int((flat_input_ids == int(image_token_id)).sum().item())
            if actual != expected:
                preview = image_grid[:8].detach().cpu().tolist()
                raise ValueError(
                    "Batch image token count mismatches image_grid_thw before model forward: "
                    f"expected={expected} actual={actual} merge_size={merge_size} "
                    f"image_grid_preview={preview} input_shape={tuple(input_ids.shape)}"
                )

    if batch_size is None or seq_len is None:
        return

    position_ids = inputs.get("position_ids")
    if isinstance(position_ids, torch.Tensor):
        if position_ids.ndim != 3 or int(position_ids.shape[0]) != 3:
            raise ValueError(
                "Batch position_ids must have shape [3, batch, seq] before model forward: "
                f"shape={tuple(position_ids.shape)}"
            )
        if int(position_ids.shape[1]) != batch_size or int(position_ids.shape[2]) != seq_len:
            raise ValueError(
                "Batch position_ids batch/seq dims mismatch input_ids before model forward: "
                f"position_ids.shape={tuple(position_ids.shape)} input_shape={tuple(input_ids.shape)}"
            )

    text_position_ids = inputs.get("text_position_ids")
    if isinstance(text_position_ids, torch.Tensor):
        if text_position_ids.ndim != 2:
            raise ValueError(
                "Batch text_position_ids must have shape [batch, seq] before model forward: "
                f"shape={tuple(text_position_ids.shape)}"
            )
        if int(text_position_ids.shape[0]) != batch_size or int(text_position_ids.shape[1]) != seq_len:
            raise ValueError(
                "Batch text_position_ids batch/seq dims mismatch input_ids before model forward: "
                f"text_position_ids.shape={tuple(text_position_ids.shape)} input_shape={tuple(input_ids.shape)}"
            )

    cu_q = inputs.get("cu_seq_lens_q")
    cu_k = inputs.get("cu_seq_lens_k")
    if isinstance(cu_q, torch.Tensor):
        if cu_q.ndim != 1:
            raise ValueError(
                "Batch cu_seq_lens_q must be a 1D tensor before model forward: "
                f"shape={tuple(cu_q.shape)}"
            )
        cu_list = [int(x) for x in cu_q.detach().cpu().tolist()]
        if not cu_list or cu_list[0] != 0 or cu_list[-1] != seq_len:
            raise ValueError(
                "Batch cu_seq_lens_q boundaries mismatch input_ids before model forward: "
                f"cu_seq_lens_q={cu_list} seq_len={seq_len}"
            )
        if any(b <= a for a, b in zip(cu_list, cu_list[1:])):
            raise ValueError(
                f"Batch cu_seq_lens_q must be strictly increasing before model forward: {cu_list}"
            )
        if isinstance(cu_k, torch.Tensor):
            cu_k_list = [int(x) for x in cu_k.detach().cpu().tolist()]
            if cu_k_list != cu_list:
                raise ValueError(
                    "Batch cu_seq_lens_k mismatch cu_seq_lens_q before model forward: "
                    f"cu_seq_lens_q={cu_list} cu_seq_lens_k={cu_k_list}"
                )
        seg_lens = [b - a for a, b in zip(cu_list, cu_list[1:])]
        max_seg_len = max(seg_lens) if seg_lens else 0
        max_length_q = inputs.get("max_length_q")
        max_length_k = inputs.get("max_length_k")
        if max_length_q is not None and int(max_length_q) != int(max_seg_len):
            raise ValueError(
                "Batch max_length_q mismatch cu_seq_lens_q before model forward: "
                f"max_length_q={int(max_length_q)} max_seg_len={int(max_seg_len)} cu_seq_lens_q={cu_list}"
            )
        if max_length_k is not None and int(max_length_k) != int(max_seg_len):
            raise ValueError(
                "Batch max_length_k mismatch cu_seq_lens_k before model forward: "
                f"max_length_k={int(max_length_k)} max_seg_len={int(max_seg_len)} cu_seq_lens_q={cu_list}"
            )
        if isinstance(text_position_ids, torch.Tensor) and batch_size == 1:
            starts = torch.nonzero(text_position_ids[0] == 0, as_tuple=False).flatten().detach().cpu().tolist()
            starts = [int(x) for x in starts]
            if starts != cu_list[:-1]:
                raise ValueError(
                    "Batch text_position_ids reset points mismatch cu_seq_lens_q before model forward: "
                    f"starts={starts[:32]} cu_seq_lens_q={cu_list[:32]}"
                )
        pack_num_samples = inputs.get("pack_num_samples")
        if isinstance(pack_num_samples, torch.Tensor) and pack_num_samples.numel() == batch_size and batch_size == 1:
            expected_segments = int(pack_num_samples.reshape(-1)[0].item())
            if expected_segments > 0 and int(len(cu_list) - 1) != expected_segments:
                raise ValueError(
                    "Batch pack_num_samples mismatch cu_seq_lens_q segment count before model forward: "
                    f"pack_num_samples={expected_segments} segment_count={int(len(cu_list)-1)} cu_seq_lens_q={cu_list}"
                )


class GradAccumLossScaleMixin:
    """CoordExp Trainer compatibility shim for packing-aware metrics.

    Background
    - We attach additional logging/diagnostic behavior without editing upstream model files.
    - For transformers>=4.57 + ms-swift, gradient-accumulation scaling is handled in the
      underlying ms-swift trainer (via the `num_items_in_batch` plumbing). We intentionally
      do NOT rescale the returned loss here to avoid double-scaling.

    Responsibilities
    - Pop packing metadata (`pack_num_samples`) and stash it on `self` for later metrics.
    - Log a few optimizer/runtime scalars as metrics so they appear in both train/eval logs.
    """

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        # Pop known batch-extras once (Stage-1 / standard SFT). They are diagnostics-only
        # fields and MUST NOT be forwarded into model(**inputs).
        from src.trainers.batch_extras import pop_and_stash_batch_extras

        pop_and_stash_batch_extras(self, inputs)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Log a few optimizer/runtime scalars as *metrics* so eval logs include them
        # (ms-swift only injects learning_rate/grad_norm into train logs by default).
        from src.metrics.reporter import SwiftMetricReporter, best_effort

        reporter = SwiftMetricReporter(self)

        def _log_runtime_metrics() -> None:
            lr_fn = getattr(self, "_get_learning_rate", None)
            if callable(lr_fn):
                reporter.update("learning_rate", float(lr_fn()))

            args = getattr(self, "args", None)
            gas = getattr(args, "gradient_accumulation_steps", None)
            if gas is not None:
                reporter.update("accum/grad_steps", float(gas))

            cur_gas = getattr(self, "current_gradient_accumulation_steps", None)
            if cur_gas is not None:
                reporter.update("accum/current_grad_steps", float(cur_gas))

        best_effort(self, name="runtime_metrics", fn=_log_runtime_metrics)

        return (loss, outputs) if return_outputs else loss


class SFTStructuralCloseLossMixin:
    """Weighted base CE for ordinary Stage-1 global CoordJSON close tokens."""

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        extras = maybe_pop_and_stash_batch_extras(self, inputs)
        cfg = getattr(self, "sft_structural_close_cfg", None)
        if cfg is None or not bool(getattr(cfg, "enabled", False)):
            return super().compute_loss(  # type: ignore[misc]
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if not isinstance(inputs, dict):
            raise TypeError("sft_structural_close enabled but inputs is not a dict")
        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "sft_structural_close enabled but inputs['labels'] is missing or not a torch.Tensor"
            )
        token_weights = getattr(extras, "sft_structural_close_token_weights", None)
        if token_weights is None or not isinstance(token_weights, torch.Tensor):
            raise ValueError(
                "custom.sft_structural_close requires sft_structural_close_token_weights from the collator"
            )
        if tuple(token_weights.shape) != tuple(labels.shape):
            raise ValueError(
                "sft_structural_close_token_weights must match labels shape"
            )

        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "sft_structural_close is enabled, but model outputs do not contain logits."
            )

        seq_len = min(int(logits.shape[1]), max(int(labels.shape[1]) - 1, 0))
        if seq_len <= 0:
            loss = logits.new_zeros(())
            return (loss, outputs) if return_outputs else loss

        logits_next = logits[:, :seq_len, :]
        labels_next = labels[:, 1 : seq_len + 1]
        weights_next = token_weights[:, 1 : seq_len + 1].to(
            device=logits.device,
            dtype=torch.float32,
        )
        labels_next = labels_next.to(device=logits.device)
        supervised = labels_next != -100
        weights_next = torch.where(
            supervised,
            weights_next,
            torch.zeros_like(weights_next),
        )

        flat_loss = F.cross_entropy(
            logits_next.float().reshape(-1, int(logits_next.shape[-1])),
            labels_next.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        flat_weights = weights_next.reshape(-1).to(dtype=flat_loss.dtype)
        denom = flat_weights.sum().clamp(min=1e-6)
        loss = torch.nan_to_num(
            (flat_loss * flat_weights).sum() / denom,
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )

        from src.metrics.reporter import SwiftMetricReporter, best_effort

        reporter = SwiftMetricReporter(self)

        def _log_structural_close_metrics() -> None:
            close_mask = supervised & weights_next.ne(1.0)
            close_count = int(close_mask.sum().detach().cpu().item())
            reporter.update(
                "sft_structural_close/final_close_weight",
                float(getattr(cfg, "final_close_weight", 1.0)),
            )
            reporter.update(
                "sft_structural_close/final_close_tokens",
                float(close_count),
            )
            reporter.update(
                "sft_structural_close/weighted_token_sum",
                float(flat_weights.sum().detach().cpu().item()),
            )
            reporter.update("loss/sft_structural_close_base_ce", float(loss.detach().cpu().item()))
            if close_count > 0:
                close_losses = flat_loss.reshape_as(labels_next)[close_mask]
                reporter.update("loss/eod", float(close_losses.mean().detach().cpu().item()))
            else:
                reporter.update("loss/eod", 0.0)

        best_effort(self, name="sft_structural_close_metrics", fn=_log_structural_close_metrics)

        compute_acc = getattr(self, "_compute_acc", None)
        if callable(compute_acc):
            best_effort(
                self,
                name="sft_structural_close_acc",
                fn=lambda: compute_acc(outputs, labels),
            )

        return (loss, outputs) if return_outputs else loss


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


class CoordSoftCEW1LossMixin:
    """Trainer mixin to compute coord-token supervision from logits.

    Behaviour when enabled:
      - Masks coord-token targets to `ignore_index` for the base full-vocab CE loss.
      - Computes coord supervision from the same forward logits (no second forward),
        by slicing logits to ordered coord-token ids (0..999) and applying:
          softCE(Gaussian soft target) + W1(CDF).
    """

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        # Ensure batch-extras are stripped before model forward.
        from src.metrics.reporter import warn_once
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        extras = None
        try:
            extras = maybe_pop_and_stash_batch_extras(self, inputs)
        except Exception:
            warn_once(
                self,
                key="batch_extras_failed",
                message=(
                    "Batch-extras extraction failed (best-effort); "
                    "continuing without extra batch diagnostics for this step."
                ),
                exc_info=True,
            )

        cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return super().compute_loss(  # type: ignore[misc]
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if not isinstance(inputs, dict):
            raise TypeError("coord_soft_ce_w1 enabled but inputs is not a dict")

        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "coord_soft_ce_w1 enabled but inputs['labels'] is missing or not a torch.Tensor"
            )

        coord_token_ids = self._get_coord_token_ids()
        if not coord_token_ids:
            raise ValueError(
                "coord_soft_ce_w1 enabled but no coord token ids found; ensure the tokenizer provides coord vocab"
            )

        labels_orig = labels
        # Use the unmasked labels for ms-swift's token_acc computation.
        setattr(self, "_coordexp_labels_for_acc", labels_orig)

        masked_labels = self._mask_coord_targets(labels_orig, coord_token_ids)
        inputs["labels"] = masked_labels

        passed_num_items = num_items_in_batch
        # Swift/Transformers may call `compute_loss(..., num_items_in_batch=...)` where
        # `num_items_in_batch` is the *number of sequences* in the micro-batch (e.g. 1),
        # not the number of supervised tokens. In that case, upstream may rescale the
        # model's per-token mean loss by (token_count / num_items_in_batch), effectively
        # turning it into a token-sum and making the logged loss depend on packing length.
        #
        # For Stage-1 (Scheme A), we want all loss terms to be mean-normalized:
        #   - base CE: mean over *non-coord* supervised tokens (coord targets are masked)
        #   - coord loss: mean over coord-token positions (handled below)
        if num_items_in_batch is not None:
            passed_num_items = self._count_supervised_tokens(masked_labels)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=passed_num_items
        )

        weighted_base_loss = None
        weighted_noncoord_sum = None
        try:
            from src.metrics.reporter import SwiftMetricReporter, best_effort

            reporter = SwiftMetricReporter(self)
            setattr(self, "_coordexp_last_weighted_base_ce", None)
            best_effort(
                self,
                name="proxy_weighted_base_ce",
                fn=lambda: self._maybe_compute_weighted_base_ce(
                    outputs=outputs,
                    masked_labels=masked_labels,
                    extras=extras,
                ),
            )
            weighted = getattr(self, "_coordexp_last_weighted_base_ce", None)
            if isinstance(weighted, tuple) and len(weighted) == 2:
                weighted_base_loss, weighted_noncoord_sum = weighted
            if isinstance(weighted_base_loss, torch.Tensor):
                loss = weighted_base_loss
            best_effort(
                self,
                name="base_ce_metrics",
                fn=lambda: self._log_base_ce_metrics(
                    reporter=reporter,
                    loss_base=loss,
                    masked_labels=masked_labels,
                    weighted_noncoord_sum=weighted_noncoord_sum,
                ),
            )

            loss = self._maybe_add_coord_softce_w1_loss(
                model=model,
                reporter=reporter,
                loss=loss,
                outputs=outputs,
                labels=labels_orig,
                masked_labels=masked_labels,
                extras=extras,
                coord_token_ids=coord_token_ids,
                num_items_in_batch=passed_num_items,
            )

            # ------------------------------------------------------------------
            # Gradient-accumulation scaling (train-only)
            # ------------------------------------------------------------------
            # When coord_soft_ce_w1 is enabled we override `num_items_in_batch` passed
            # to ms-swift to enforce packing-safe mean normalization for the base CE.
            #
            # That override prevents transformers>=4.57 from applying its usual
            # gradient-accumulation scaling (it only divides when num_items_in_batch is None).
            # To keep `loss` and `eval_loss` on the same scale (and keep gradients stable),
            # we manually divide by the current accumulation steps here.
            #
            # NOTE: we only do this in TRAIN mode, and only when the outer trainer did pass
            # a non-None `num_items_in_batch` (i.e. we're in the 4.57+ scaling path).
            if getattr(self, "model", None) is not None and bool(self.model.training):
                if (
                    bool(getattr(self, "model_accepts_loss_kwargs", False))
                    and num_items_in_batch is not None
                    and getattr(self, "compute_loss_func", None) is None
                ):
                    gas = getattr(self, "current_gradient_accumulation_steps", None)
                    if gas is None:
                        args = getattr(self, "args", None)
                        gas = getattr(args, "gradient_accumulation_steps", None)
                    gas_int = int(gas or 1)
                    if gas_int > 1:
                        loss = loss / float(gas_int)
        finally:
            inputs["labels"] = labels_orig
            setattr(self, "_coordexp_labels_for_acc", None)

        return (loss, outputs) if return_outputs else loss

    def _maybe_compute_weighted_base_ce(
        self,
        *,
        outputs: Any,
        masked_labels: torch.Tensor,
        extras: Any,
    ) -> None:
        logits = getattr(outputs, "logits", None)
        token_types = getattr(extras, "token_types", None) if extras is not None else None
        desc_weights = (
            getattr(extras, "proxy_desc_token_weights", None)
            if extras is not None
            else None
        )
        if (
            not isinstance(logits, torch.Tensor)
            or not isinstance(token_types, torch.Tensor)
            or not isinstance(desc_weights, torch.Tensor)
        ):
            setattr(self, "_coordexp_last_weighted_base_ce", None)
            return
        if tuple(token_types.shape) != tuple(masked_labels.shape):
            setattr(self, "_coordexp_last_weighted_base_ce", None)
            return
        if tuple(desc_weights.shape) != tuple(masked_labels.shape):
            setattr(self, "_coordexp_last_weighted_base_ce", None)
            return

        seq_len = min(int(logits.shape[1]), max(int(masked_labels.shape[1]) - 1, 0))
        if seq_len <= 0:
            setattr(self, "_coordexp_last_weighted_base_ce", None)
            return

        logits_next = logits[:, :seq_len, :]
        labels_next = masked_labels[:, 1 : seq_len + 1]
        token_types_next = token_types[:, 1 : seq_len + 1]
        desc_weights_next = desc_weights[:, 1 : seq_len + 1].to(dtype=torch.float32)

        supervised = labels_next != -100
        base_weights = supervised.to(dtype=torch.float32)
        desc_mask = supervised & (token_types_next == TokenType.DESC)
        base_weights[desc_mask] = desc_weights_next[desc_mask]

        flat_logits = logits_next.reshape(-1, int(logits_next.shape[-1]))
        flat_labels = labels_next.reshape(-1)
        flat_weights = base_weights.reshape(-1)
        rows_per_chunk = 4096
        n_rows = int(flat_labels.numel())
        ce_num = None
        for start in range(0, n_rows, rows_per_chunk):
            end = min(start + rows_per_chunk, n_rows)
            ce_chunk = F.cross_entropy(
                flat_logits[start:end].float(),
                flat_labels[start:end],
                ignore_index=-100,
                reduction="none",
            )
            if ce_num is None:
                ce_num = ce_chunk.new_tensor(0.0)
            ce_num = ce_num + (
                ce_chunk * flat_weights[start:end].to(dtype=ce_chunk.dtype)
            ).sum()
        if ce_num is None:
            setattr(self, "_coordexp_last_weighted_base_ce", None)
            return

        denom = flat_weights.sum().to(dtype=ce_num.dtype).clamp(min=1e-6)
        weighted_loss = torch.nan_to_num(
            ce_num / denom,
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )
        setattr(
            self,
            "_coordexp_last_weighted_base_ce",
            (
                weighted_loss,
                float(flat_weights.sum().detach().cpu().item()),
            ),
        )

    def _compute_acc(self, outputs, labels, cu_seqlens=None) -> None:
        """Force ms-swift token_acc to use unmasked labels (incl. coord tokens).

        coord_soft_ce_w1 masks coord-token targets to -100 for the base CE loss.
        For reporting consistency, token_acc should always be computed on the original
        supervised labels (labels != -100), including coord-token positions.

        Note: The current pinned ms-swift (swift==3.10.0.dev0) implements
        `SwiftMixin._compute_acc(self, outputs, labels)` without a `cu_seqlens` kwarg.
        Some call sites may still pass `cu_seqlens`; we accept it but intentionally
        *do not forward it*.
        """

        labels_for_acc = getattr(self, "_coordexp_labels_for_acc", None)
        if isinstance(labels_for_acc, torch.Tensor):
            labels = labels_for_acc

        return super()._compute_acc(outputs, labels)

    def _log_base_ce_metrics(
        self,
        *,
        reporter: Any,
        loss_base: torch.Tensor,
        masked_labels: torch.Tensor,
        weighted_noncoord_sum: float | None = None,
    ) -> None:
        """Log the base CE (non-coord) component, so train/eval loss parts line up."""

        reporter.update("base_ce/loss", float(loss_base.detach().cpu().item()))

        noncoord_tokens = int((masked_labels[:, 1:] != -100).sum().detach().item())
        reporter.update("base_ce/noncoord_tokens", float(noncoord_tokens))
        if weighted_noncoord_sum is not None:
            reporter.update("base_ce/noncoord_weight_sum", float(weighted_noncoord_sum))

        # Per-sample normalization for packed runs: interpret a "unit" as a pack of N samples.
        # This is a logging-only helper (does not affect optimization).
        total_samples = None
        pack_n = getattr(self, "_coordexp_pack_num_samples", None)
        if isinstance(pack_n, torch.Tensor):
            total_samples = float(pack_n.detach().sum().cpu().item())
        elif isinstance(pack_n, (list, tuple)):
            try:
                total_samples = float(sum(int(v) for v in pack_n))
            except (TypeError, ValueError):
                total_samples = None
        elif isinstance(pack_n, (int, float)):
            total_samples = float(pack_n)
        if total_samples is None:
            total_samples = float(masked_labels.shape[0])
        total_samples = max(1.0, float(total_samples))

        reporter.update("pack/num_samples", float(total_samples))
        reporter.update(
            "base_ce/noncoord_tokens_per_sample",
            float(noncoord_tokens) / float(total_samples),
        )

        loss_per_sample = (
            float(loss_base.detach().cpu().item())
            * float(noncoord_tokens)
            / float(total_samples)
        )
        reporter.update("base_ce/loss_per_sample", float(loss_per_sample))

        # Stash for the stage1 total-per-sample estimate (logged from coord loss block).
        setattr(self, "_coordexp_last_base_loss_per_sample", float(loss_per_sample))

    def _maybe_add_coord_softce_w1_loss(
        self,
        *,
        model: Any,
        reporter: Any,
        loss: torch.Tensor,
        outputs: Any,
        labels: torch.Tensor,
        masked_labels: torch.Tensor,
        extras: Any,
        coord_token_ids: list[int],
        num_items_in_batch: Any,
    ) -> torch.Tensor:
        cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return loss

        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "coord_soft_ce_w1 is enabled, but model outputs do not contain logits."
            )

        vocab_size = int(logits.shape[-1])
        coord_id_map = self._get_coord_id_map(vocab_size, logits.device)
        if coord_id_map is None:
            raise RuntimeError(
                "coord_soft_ce_w1 is enabled, but coord_id_map could not be constructed."
            )

        avg_tokens = bool(
            getattr(getattr(self, "args", None), "average_tokens_across_devices", False)
        )
        model_accepts = bool(getattr(self, "model_accepts_loss_kwargs", False))
        acc_num_proc = None
        try:
            acc = getattr(self, "accelerator", None)
            acc_num_proc = int(getattr(acc, "num_processes", 0) or 0)
        except (TypeError, ValueError):
            acc_num_proc = None

        from src.trainers.losses.coord_soft_ce_w1 import compute_coord_soft_ce_w1_loss

        result = compute_coord_soft_ce_w1_loss(
            logits=logits,
            labels=labels,
            masked_labels=masked_labels,
            coord_token_weights=getattr(extras, "proxy_coord_token_weights", None),
            coord_token_ids=coord_token_ids,
            coord_id_map=coord_id_map,
            tokenizer=getattr(getattr(self, "template", None), "tokenizer", None),
            token_types=getattr(extras, "token_types", None),
            cfg=cfg,
            average_tokens_across_devices=avg_tokens,
            model_accepts_loss_kwargs=model_accepts,
            accelerator_num_processes=acc_num_proc,
            object_field_order=str(
                getattr(self, "object_field_order", "desc_first") or "desc_first"
            ),
            bbox_format=str(getattr(self, "bbox_format", "xyxy") or "xyxy"),
        )

        if result is None:
            return loss

        self._log_coord_softce_w1_metrics(
            reporter=reporter,
            result=result,
            batch_size=int(labels.shape[0]),
        )

        from src.metrics.reporter import best_effort_value
        from src.trainers.monitoring.loss_gradient_monitor import (
            build_stage1_coord_monitor_terms,
            get_loss_gradient_monitor,
        )

        monitor = get_loss_gradient_monitor(self)
        if monitor is not None:
            gradmon_metrics = best_effort_value(
                self,
                name="loss_gradient_monitor",
                fn=lambda: monitor.measure(
                    model=model,
                    loss_terms=build_stage1_coord_monitor_terms(result=result, cfg=cfg),
                ),
                default={},
            )
            if isinstance(gradmon_metrics, Mapping) and gradmon_metrics:
                reporter.update_many(gradmon_metrics)

        return loss + result.coord_loss.to(dtype=loss.dtype)

    def _log_coord_softce_w1_metrics(
        self,
        *,
        reporter: Any,
        result: Any,
        batch_size: int,
    ) -> None:
        """Log coord_softce_w1 + coord_diag metrics (diagnostics-only)."""

        loss_total = getattr(result, "coord_loss", None)
        loss_softce = getattr(result, "softce_contrib", None)
        loss_w1 = getattr(result, "w1_contrib", None)
        loss_ce = getattr(result, "ce_contrib", None)
        loss_gate = getattr(result, "gate_contrib", None)
        loss_text_gate = getattr(result, "text_gate_contrib", None)

        if not isinstance(loss_total, torch.Tensor):
            return

        reporter.update("coord_softce_w1/loss", float(loss_total.detach().cpu().item()))
        if isinstance(loss_softce, torch.Tensor):
            reporter.update(
                "coord_softce_w1/soft_ce", float(loss_softce.detach().cpu().item())
            )
        if isinstance(loss_w1, torch.Tensor):
            reporter.update("coord_softce_w1/w1", float(loss_w1.detach().cpu().item()))
        if isinstance(loss_ce, torch.Tensor):
            reporter.update("coord_softce_w1/ce", float(loss_ce.detach().cpu().item()))
        if isinstance(loss_gate, torch.Tensor):
            reporter.update("coord_softce_w1/gate", float(loss_gate.detach().cpu().item()))
        if isinstance(loss_text_gate, torch.Tensor):
            reporter.update(
                "coord_softce_w1/text_gate",
                float(loss_text_gate.detach().cpu().item()),
            )
        loss_adjacent = getattr(result, "adjacent_repulsion_contrib", None)
        if isinstance(loss_adjacent, torch.Tensor):
            reporter.update(
                "coord_softce_w1/adjacent_repulsion",
                float(loss_adjacent.detach().cpu().item()),
            )

        coord_tokens = int(getattr(result, "coord_tokens", 0) or 0)

        # Stable tags across loss modes (pure CE vs softCE+W1+gate).
        reporter.update("coord_diag/enabled", 1.0)
        reporter.update("coord_diag/loss", float(loss_total.detach().cpu().item()))
        if isinstance(loss_softce, torch.Tensor):
            reporter.update("coord_diag/soft_ce", float(loss_softce.detach().cpu().item()))
        if isinstance(loss_w1, torch.Tensor):
            reporter.update("coord_diag/w1", float(loss_w1.detach().cpu().item()))
        if isinstance(loss_ce, torch.Tensor):
            reporter.update("coord_diag/ce", float(loss_ce.detach().cpu().item()))
        if isinstance(loss_gate, torch.Tensor):
            reporter.update("coord_diag/gate", float(loss_gate.detach().cpu().item()))
        if isinstance(loss_text_gate, torch.Tensor):
            reporter.update(
                "coord_diag/text_gate",
                float(loss_text_gate.detach().cpu().item()),
            )
        if isinstance(loss_adjacent, torch.Tensor):
            reporter.update(
                "coord_diag/adjacent_repulsion",
                float(loss_adjacent.detach().cpu().item()),
            )
        reporter.update("coord_diag/coord_tokens", float(coord_tokens))
        reporter.update(
            "coord_diag/adjacent_repulsion_pair_count",
            float(int(getattr(result, "adjacent_repulsion_pair_count", 0) or 0)),
        )
        reporter.update(
            "coord_diag/adjacent_repulsion_applied_count",
            float(int(getattr(result, "adjacent_repulsion_applied_count", 0) or 0)),
        )
        copy_score_mean = getattr(result, "adjacent_repulsion_copy_score_mean", None)
        if isinstance(copy_score_mean, torch.Tensor):
            reporter.update(
                "coord_diag/adjacent_repulsion_copy_score_mean",
                float(copy_score_mean.detach().cpu().item()),
            )

        gate_mass_mean = getattr(result, "gate_mass_mean", None)
        if isinstance(gate_mass_mean, torch.Tensor):
            reporter.update(
                "coord_diag/coord_vocab_mass", float(gate_mass_mean.detach().cpu().item())
            )
        text_gate_coord_mass_mean = getattr(result, "text_gate_coord_mass_mean", None)
        if isinstance(text_gate_coord_mass_mean, torch.Tensor):
            reporter.update(
                "coord_diag/text_coord_vocab_mass",
                float(text_gate_coord_mass_mean.detach().cpu().item()),
            )

        coord_acc_top5 = getattr(result, "coord_acc_top5", None)
        if isinstance(coord_acc_top5, torch.Tensor):
            reporter.update("coord_diag/acc_top5", float(coord_acc_top5.detach().cpu().item()))

        coord_p_gt_mean = getattr(result, "coord_p_gt_mean", None)
        if isinstance(coord_p_gt_mean, torch.Tensor):
            reporter.update("coord_diag/p_gt_mean", float(coord_p_gt_mean.detach().cpu().item()))

        coord_margin_mean = getattr(result, "coord_margin_mean", None)
        if isinstance(coord_margin_mean, torch.Tensor):
            reporter.update(
                "coord_diag/margin_mean", float(coord_margin_mean.detach().cpu().item())
            )

        coord_expected_bin_mae = getattr(result, "coord_expected_bin_mae", None)
        if isinstance(coord_expected_bin_mae, torch.Tensor):
            reporter.update(
                "coord_diag/expected_bin_mae", float(coord_expected_bin_mae.detach().cpu().item())
            )

        coord_expected_bin_abs_err_p90 = getattr(result, "coord_expected_bin_abs_err_p90", None)
        if isinstance(coord_expected_bin_abs_err_p90, torch.Tensor):
            reporter.update(
                "coord_diag/expected_bin_abs_err_p90",
                float(coord_expected_bin_abs_err_p90.detach().cpu().item()),
            )

        coord_w1_to_delta = getattr(result, "coord_w1_to_delta", None)
        if isinstance(coord_w1_to_delta, torch.Tensor):
            reporter.update("coord_diag/w1_to_delta", float(coord_w1_to_delta.detach().cpu().item()))

        # Per-sample normalization (packed units).
        total_samples = None
        pack_n = getattr(self, "_coordexp_pack_num_samples", None)
        if isinstance(pack_n, torch.Tensor):
            total_samples = float(pack_n.detach().sum().cpu().item())
        elif isinstance(pack_n, (list, tuple)):
            total_samples = float(sum(int(v) for v in pack_n))
        elif isinstance(pack_n, (int, float)):
            total_samples = float(pack_n)
        if total_samples is None:
            total_samples = float(batch_size)
        total_samples = max(1.0, float(total_samples))

        reporter.update(
            "coord_diag/coord_tokens_per_sample",
            float(coord_tokens) / float(total_samples),
        )
        coord_loss_per_sample = (
            float(loss_total.detach().cpu().item()) * float(coord_tokens) / float(total_samples)
        )
        reporter.update("coord_diag/loss_per_sample", float(coord_loss_per_sample))
        setattr(self, "_coordexp_last_coord_loss_per_sample", float(coord_loss_per_sample))

        base_loss_per_sample = getattr(self, "_coordexp_last_base_loss_per_sample", None)
        if isinstance(base_loss_per_sample, (int, float)):
            reporter.update(
                "stage1/total_loss_per_sample_est",
                float(base_loss_per_sample) + float(coord_loss_per_sample),
            )

    def _coord_vocab_gate_loss(
        self,
        logits_full: torch.Tensor,
        logits_coord: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        """Negative log probability mass of the coord sub-vocabulary.

        This delegates to the shared helper used by loss and rollout paths so all
        consumers share identical numeric fences.
        """

        from src.trainers.losses.coord_soft_ce_w1 import coord_vocab_gate_loss

        gate, _mass_mean = coord_vocab_gate_loss(
            logits_full=logits_full,
            logits_coord=logits_coord,
            temperature=float(temperature),
        )
        return gate

    def _count_supervised_tokens(self, labels: torch.Tensor) -> int:
        from src.trainers.losses.coord_soft_ce_w1 import count_supervised_tokens

        return count_supervised_tokens(labels)

    def _mask_coord_targets(
        self, labels: torch.Tensor, coord_token_ids: list[int]
    ) -> torch.Tensor:
        from src.trainers.teacher_forcing.stage1 import mask_stage1_coord_targets

        return mask_stage1_coord_targets(labels, coord_token_ids)

    def _get_coord_token_ids(self) -> list[int]:
        cached = getattr(self, "_coord_token_ids", None)
        if cached is not None:
            return cached
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        if tokenizer is None:
            return []
        ids = get_coord_token_ids(tokenizer, validate=True)
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

        from src.trainers.losses.coord_soft_ce_w1 import build_coord_id_map

        id_map = build_coord_id_map(
            vocab_size=int(vocab_size), device=device, coord_token_ids=coord_token_ids
        )
        cache[key] = id_map
        return id_map


class BBoxGeoLossMixin:
    """Stage-1 bbox-geometry aux loss host.

    This keeps the standard Stage-1 SFT trainer and adds decoded-box geometry
    supervision from the same forward logits used by the coord-token losses.
    """

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        cfg = getattr(self, "bbox_geo_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return super().compute_loss(  # type: ignore[misc]
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if not isinstance(inputs, dict):
            raise TypeError("bbox_geo enabled but inputs is not a dict")

        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "bbox_geo enabled but inputs['labels'] is missing or not a torch.Tensor"
            )

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "bbox_geo is enabled, but model outputs do not contain logits."
            )

        coord_token_ids = self._bbox_geo_coord_token_ids()
        if not coord_token_ids:
            raise RuntimeError(
                "bbox_geo enabled but no coord token ids found; ensure the tokenizer provides coord vocab"
            )

        coord_id_map = self._bbox_geo_coord_id_map(
            vocab_size=int(logits.shape[-1]),
            device=logits.device,
        )
        if coord_id_map is None:
            raise RuntimeError(
                "bbox_geo is enabled, but coord_id_map could not be constructed."
            )

        from src.metrics.reporter import SwiftMetricReporter, best_effort_value
        from src.trainers.losses.bbox_geo import compute_stage1_bbox_geo_loss
        from src.trainers.monitoring.loss_gradient_monitor import (
            build_stage1_bbox_geo_monitor_terms,
            get_loss_gradient_monitor,
        )

        coord_cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
        decode_temperature = 1.0
        if coord_cfg is not None:
            try:
                decode_temperature = float(getattr(coord_cfg, "temperature", 1.0) or 1.0)
            except (TypeError, ValueError):
                decode_temperature = 1.0

        result = compute_stage1_bbox_geo_loss(
            logits=logits,
            labels=labels,
            coord_token_ids=coord_token_ids,
            coord_id_map=coord_id_map,
            tokenizer=getattr(getattr(self, "template", None), "tokenizer", None),
            cfg=cfg,
            decode_temperature=float(max(1e-6, decode_temperature)),
            decode_mode="exp",
            object_field_order=str(
                getattr(self, "object_field_order", "desc_first") or "desc_first"
            ),
            bbox_format=str(getattr(self, "bbox_format", "xyxy") or "xyxy"),
        )
        if result is None:
            return (loss, outputs) if return_outputs else loss

        reporter = SwiftMetricReporter(self)
        self._log_bbox_geo_metrics(
            reporter=reporter,
            result=result,
            batch_size=int(labels.shape[0]),
        )

        monitor = get_loss_gradient_monitor(self)
        if monitor is not None:
            gradmon_metrics = best_effort_value(
                self,
                name="loss_gradient_monitor",
                fn=lambda: monitor.measure(
                    model=model,
                    loss_terms=build_stage1_bbox_geo_monitor_terms(result=result, cfg=cfg),
                ),
                default={},
            )
            if isinstance(gradmon_metrics, Mapping) and gradmon_metrics:
                reporter.update_many(gradmon_metrics)

        loss = loss + result.total_loss.to(dtype=loss.dtype)
        return (loss, outputs) if return_outputs else loss

    def _log_bbox_geo_metrics(
        self,
        *,
        reporter: Any,
        result: Any,
        batch_size: int,
    ) -> None:
        total_loss = getattr(result, "total_loss", None)
        if not isinstance(total_loss, torch.Tensor):
            return

        smoothl1 = getattr(result, "smoothl1_contrib", None)
        ciou = getattr(result, "ciou_contrib", None)
        bbox_groups = int(getattr(result, "bbox_groups", 0) or 0)
        coord_slots = int(getattr(result, "coord_slots", 0) or 0)
        skipped_rows = int(getattr(result, "skipped_incomplete_rows", 0) or 0)
        skipped_coord_slots = int(
            getattr(result, "skipped_incomplete_coord_slots", 0) or 0
        )

        reporter.update("loss/geo/bbox_geo", float(total_loss.detach().cpu().item()))
        if isinstance(smoothl1, torch.Tensor):
            reporter.update("loss/geo/bbox_smoothl1", float(smoothl1.detach().cpu().item()))
        if isinstance(ciou, torch.Tensor):
            reporter.update("loss/geo/bbox_ciou", float(ciou.detach().cpu().item()))

        reporter.update("bbox_geo/groups_total", float(bbox_groups))
        reporter.update("bbox_geo/coord_slots_total", float(coord_slots))
        reporter.update("bbox_geo/skipped_incomplete_rows", float(skipped_rows))
        reporter.update(
            "bbox_geo/skipped_incomplete_coord_slots", float(skipped_coord_slots)
        )

        total_samples = None
        pack_n = getattr(self, "_coordexp_pack_num_samples", None)
        if isinstance(pack_n, torch.Tensor):
            total_samples = float(pack_n.detach().sum().cpu().item())
        elif isinstance(pack_n, (list, tuple)):
            total_samples = float(sum(int(v) for v in pack_n))
        elif isinstance(pack_n, (int, float)):
            total_samples = float(pack_n)
        if total_samples is None:
            total_samples = float(batch_size)
        total_samples = max(1.0, float(total_samples))

        reporter.update(
            "bbox_geo/groups_per_sample",
            float(bbox_groups) / float(total_samples),
        )
        bbox_loss_per_sample = (
            float(total_loss.detach().cpu().item()) * float(bbox_groups) / float(total_samples)
        )
        reporter.update("bbox_geo/loss_per_sample", float(bbox_loss_per_sample))
        setattr(self, "_coordexp_last_bbox_geo_loss_per_sample", float(bbox_loss_per_sample))

        base_loss_per_sample = getattr(self, "_coordexp_last_base_loss_per_sample", None)
        coord_loss_per_sample = getattr(self, "_coordexp_last_coord_loss_per_sample", None)
        total_est = 0.0
        has_total = False
        if isinstance(base_loss_per_sample, (int, float)):
            total_est += float(base_loss_per_sample)
            has_total = True
        if isinstance(coord_loss_per_sample, (int, float)):
            total_est += float(coord_loss_per_sample)
            has_total = True
        bbox_geo_loss_per_sample = getattr(self, "_coordexp_last_bbox_geo_loss_per_sample", None)
        if isinstance(bbox_geo_loss_per_sample, (int, float)):
            total_est += float(bbox_geo_loss_per_sample)
            has_total = True
        if has_total:
            reporter.update(
                "stage1/total_loss_per_sample_est",
                float(total_est) + float(bbox_loss_per_sample),
            )

    def _bbox_geo_coord_token_ids(self) -> list[int]:
        coord_ids_fn = getattr(self, "_get_coord_token_ids", None)
        if callable(coord_ids_fn):
            coord_token_ids = coord_ids_fn()
            if coord_token_ids:
                return coord_token_ids
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        if tokenizer is None:
            return []
        return get_coord_token_ids(tokenizer, validate=True)

    def _bbox_geo_coord_id_map(
        self, vocab_size: int, device: torch.device
    ) -> torch.Tensor | None:
        coord_map_fn = getattr(self, "_get_coord_id_map", None)
        if callable(coord_map_fn):
            out = coord_map_fn(vocab_size, device)
            if isinstance(out, torch.Tensor):
                return out

        coord_token_ids = self._bbox_geo_coord_token_ids()
        if not coord_token_ids:
            return None

        from src.trainers.losses.coord_soft_ce_w1 import build_coord_id_map

        return build_coord_id_map(
            vocab_size=int(vocab_size),
            device=device,
            coord_token_ids=coord_token_ids,
        )


class BBoxSizeAuxLossMixin:
    """Stage-1 bbox-size auxiliary loss host (bbox-only v1).

    This wraps the existing single forward and reuses the shared decoded-box
    size helper used by Stage-2 teacher-forcing plugins.
    """

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        cfg = getattr(self, "bbox_size_aux_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return super().compute_loss(  # type: ignore[misc]
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if not isinstance(inputs, dict):
            raise TypeError("bbox_size_aux enabled but inputs is not a dict")

        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "bbox_size_aux enabled but inputs['labels'] is missing or not a torch.Tensor"
            )

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "bbox_size_aux is enabled, but model outputs do not contain logits."
            )

        coord_token_ids = self._bbox_size_aux_coord_token_ids()
        if not coord_token_ids:
            raise RuntimeError(
                "bbox_size_aux enabled but no coord token ids found; ensure the tokenizer provides coord vocab"
            )

        coord_id_map = self._bbox_size_aux_coord_id_map(
            vocab_size=int(logits.shape[-1]),
            device=logits.device,
        )
        if coord_id_map is None:
            raise RuntimeError(
                "bbox_size_aux is enabled, but coord_id_map could not be constructed."
            )

        from src.metrics.reporter import SwiftMetricReporter, best_effort_value
        from src.trainers.losses.bbox_size_aux import compute_stage1_bbox_size_aux_loss
        from src.trainers.monitoring.loss_gradient_monitor import (
            build_stage1_bbox_size_monitor_terms,
            get_loss_gradient_monitor,
        )

        coord_cfg = getattr(self, "coord_soft_ce_w1_cfg", None)
        decode_temperature = 1.0
        if coord_cfg is not None:
            try:
                decode_temperature = float(getattr(coord_cfg, "temperature", 1.0) or 1.0)
            except (TypeError, ValueError):
                decode_temperature = 1.0

        result = compute_stage1_bbox_size_aux_loss(
            logits=logits,
            labels=labels,
            coord_token_ids=coord_token_ids,
            coord_id_map=coord_id_map,
            tokenizer=getattr(getattr(self, "template", None), "tokenizer", None),
            cfg=cfg,
            decode_temperature=float(max(1e-6, decode_temperature)),
            decode_mode="exp",
            object_field_order=str(
                getattr(self, "object_field_order", "desc_first") or "desc_first"
            ),
            bbox_format=str(getattr(self, "bbox_format", "xyxy") or "xyxy"),
        )
        if result is None:
            return (loss, outputs) if return_outputs else loss

        reporter = SwiftMetricReporter(self)
        self._log_bbox_size_aux_metrics(
            reporter=reporter,
            result=result,
            batch_size=int(labels.shape[0]),
        )

        monitor = get_loss_gradient_monitor(self)
        if monitor is not None:
            gradmon_metrics = best_effort_value(
                self,
                name="loss_gradient_monitor",
                fn=lambda: monitor.measure(
                    model=model,
                    loss_terms=build_stage1_bbox_size_monitor_terms(result=result, cfg=cfg),
                ),
                default={},
            )
            if isinstance(gradmon_metrics, Mapping) and gradmon_metrics:
                reporter.update_many(gradmon_metrics)

        loss = loss + result.total_loss.to(dtype=loss.dtype)
        return (loss, outputs) if return_outputs else loss

    def _log_bbox_size_aux_metrics(
        self,
        *,
        reporter: Any,
        result: Any,
        batch_size: int,
    ) -> None:
        total_loss = getattr(result, "total_loss", None)
        if not isinstance(total_loss, torch.Tensor):
            return

        log_wh = getattr(result, "log_wh_contrib", None)
        oversize = getattr(result, "oversize_contrib", None)
        stats = getattr(result, "stats", None)
        bbox_groups = int(getattr(result, "bbox_groups", 0) or 0)
        coord_slots = int(getattr(result, "coord_slots", 0) or 0)
        skipped_rows = int(getattr(result, "skipped_incomplete_rows", 0) or 0)
        skipped_coord_slots = int(
            getattr(result, "skipped_incomplete_coord_slots", 0) or 0
        )

        reporter.update("loss/geo/bbox_size_aux", float(total_loss.detach().cpu().item()))
        if isinstance(log_wh, torch.Tensor):
            reporter.update("loss/geo/bbox_log_wh", float(log_wh.detach().cpu().item()))
        if isinstance(oversize, torch.Tensor):
            reporter.update("loss/geo/bbox_oversize", float(oversize.detach().cpu().item()))

        reporter.update("bbox_size_aux/groups_total", float(bbox_groups))
        reporter.update("bbox_size_aux/coord_slots_total", float(coord_slots))
        reporter.update(
            "bbox_size_aux/skipped_incomplete_rows", float(skipped_rows)
        )
        reporter.update(
            "bbox_size_aux/skipped_incomplete_coord_slots",
            float(skipped_coord_slots),
        )
        if stats is not None:
            reporter.update(
                "bbox_size_aux/mean_width",
                float(stats.mean_width.detach().cpu().item()),
            )
            reporter.update(
                "bbox_size_aux/mean_height",
                float(stats.mean_height.detach().cpu().item()),
            )
            reporter.update(
                "bbox_size_aux/mean_log_area",
                float(stats.mean_log_area.detach().cpu().item()),
            )

        total_samples = None
        pack_n = getattr(self, "_coordexp_pack_num_samples", None)
        if isinstance(pack_n, torch.Tensor):
            total_samples = float(pack_n.detach().sum().cpu().item())
        elif isinstance(pack_n, (list, tuple)):
            total_samples = float(sum(int(v) for v in pack_n))
        elif isinstance(pack_n, (int, float)):
            total_samples = float(pack_n)
        if total_samples is None:
            total_samples = float(batch_size)
        total_samples = max(1.0, float(total_samples))

        reporter.update(
            "bbox_size_aux/groups_per_sample",
            float(bbox_groups) / float(total_samples),
        )
        bbox_loss_per_sample = (
            float(total_loss.detach().cpu().item()) * float(bbox_groups) / float(total_samples)
        )
        reporter.update("bbox_size_aux/loss_per_sample", float(bbox_loss_per_sample))
        setattr(self, "_coordexp_last_bbox_size_loss_per_sample", float(bbox_loss_per_sample))

        base_loss_per_sample = getattr(self, "_coordexp_last_base_loss_per_sample", None)
        coord_loss_per_sample = getattr(self, "_coordexp_last_coord_loss_per_sample", None)
        bbox_geo_loss_per_sample = getattr(self, "_coordexp_last_bbox_geo_loss_per_sample", None)
        total_est = 0.0
        has_total = False
        if isinstance(base_loss_per_sample, (int, float)):
            total_est += float(base_loss_per_sample)
            has_total = True
        if isinstance(coord_loss_per_sample, (int, float)):
            total_est += float(coord_loss_per_sample)
            has_total = True
        if isinstance(bbox_geo_loss_per_sample, (int, float)):
            total_est += float(bbox_geo_loss_per_sample)
            has_total = True
        if has_total:
            reporter.update(
                "stage1/total_loss_per_sample_est",
                float(total_est) + float(bbox_loss_per_sample),
            )

    def _bbox_size_aux_coord_token_ids(self) -> list[int]:
        coord_ids_fn = getattr(self, "_get_coord_token_ids", None)
        if callable(coord_ids_fn):
            coord_token_ids = coord_ids_fn()
            if coord_token_ids:
                return coord_token_ids
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        if tokenizer is None:
            return []
        return get_coord_token_ids(tokenizer, validate=True)

    def _bbox_size_aux_coord_id_map(
        self, vocab_size: int, device: torch.device
    ) -> torch.Tensor | None:
        coord_map_fn = getattr(self, "_get_coord_id_map", None)
        if callable(coord_map_fn):
            out = coord_map_fn(vocab_size, device)
            if isinstance(out, torch.Tensor):
                return out

        coord_token_ids = self._bbox_size_aux_coord_token_ids()
        if not coord_token_ids:
            return None

        from src.trainers.losses.coord_soft_ce_w1 import build_coord_id_map

        return build_coord_id_map(
            vocab_size=int(vocab_size),
            device=device,
            coord_token_ids=coord_token_ids,
        )
