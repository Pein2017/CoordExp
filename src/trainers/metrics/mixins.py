from typing import Any

import torch
import torch.distributed as dist

from src.coord_tokens.codec import get_coord_token_ids
from src.data_collators.token_types import TokenType


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


class AggregateTokenTypeMetricsMixin:
    """Trainer mixin to log aggregate loss/accuracy and token-type metrics.

    - Aggregate only (no per-dataset buckets)
    - Safe under packing when token_types are pre-concatenated; skips on mismatch
    - Skips metrics when no supervised tokens to avoid NaNs

    Metric key reference:
      - docs/training/METRICS_LOSSES.md
    """

    label_field = "dataset_labels"
    segment_field = "dataset_segments"

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Ensure batch-extras are stripped before model forward (Stage-1).
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        extras = maybe_pop_and_stash_batch_extras(self, inputs)
        token_types = extras.token_types

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
        if not dist.is_available() or not dist.is_initialized():
            return
        mode = (
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
        )  # type: ignore[attr-defined]
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
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        maybe_pop_and_stash_batch_extras(self, inputs)

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

        try:
            from src.metrics.reporter import SwiftMetricReporter, best_effort

            reporter = SwiftMetricReporter(self)
            best_effort(
                self,
                name="base_ce_metrics",
                fn=lambda: self._log_base_ce_metrics(
                    reporter=reporter,
                    loss_base=loss,
                    masked_labels=masked_labels,
                ),
            )

            loss = self._maybe_add_coord_softce_w1_loss(
                reporter=reporter,
                loss=loss,
                outputs=outputs,
                labels=labels_orig,
                masked_labels=masked_labels,
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
        self, *, reporter: Any, loss_base: torch.Tensor, masked_labels: torch.Tensor
    ) -> None:
        """Log the base CE (non-coord) component, so train/eval loss parts line up."""

        reporter.update("base_ce/loss", float(loss_base.detach().cpu().item()))

        noncoord_tokens = int((masked_labels[:, 1:] != -100).sum().detach().item())
        reporter.update("base_ce/noncoord_tokens", float(noncoord_tokens))

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
        reporter: Any,
        loss: torch.Tensor,
        outputs: Any,
        labels: torch.Tensor,
        masked_labels: torch.Tensor,
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
            coord_token_ids=coord_token_ids,
            coord_id_map=coord_id_map,
            cfg=cfg,
            average_tokens_across_devices=avg_tokens,
            model_accepts_loss_kwargs=model_accepts,
            accelerator_num_processes=acc_num_proc,
        )

        if result is None:
            return loss

        self._log_coord_softce_w1_metrics(
            reporter=reporter,
            result=result,
            batch_size=int(labels.shape[0]),
        )

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
        reporter.update("coord_diag/coord_tokens", float(coord_tokens))

        gate_mass_mean = getattr(result, "gate_mass_mean", None)
        if isinstance(gate_mass_mean, torch.Tensor):
            reporter.update(
                "coord_diag/coord_vocab_mass", float(gate_mass_mean.detach().cpu().item())
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
        from src.trainers.losses.coord_soft_ce_w1 import mask_coord_targets

        return mask_coord_targets(labels, coord_token_ids)

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



# Kept here for backward compatibility; implementation lives in trainers/monitoring.
from src.trainers.monitoring.instability import InstabilityMonitorMixin
