from typing import Any, Mapping

import torch
import torch.nn.functional as F

from src.coord_tokens.codec import get_coord_token_ids
from src.data_collators.token_types import TokenType


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

        return loss.float() + result.coord_loss.float()

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
        reporter.update("coord_diag/coord_tokens", float(coord_tokens))

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


__all__ = [
    "CoordSoftCEW1LossMixin",
]
