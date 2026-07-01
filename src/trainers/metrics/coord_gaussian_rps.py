from __future__ import annotations

from typing import Any

import torch

from src.coord_tokens.codec import get_coord_token_ids


class CoordGaussianRPSLossMixin:
    """Trainer mixin for ordinary Stage-1 Gaussian + RPS coordinate loss."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        from src.metrics.reporter import SwiftMetricReporter, warn_once
        from src.trainers.batch_extras import maybe_pop_and_stash_batch_extras

        extras = None
        try:
            extras = maybe_pop_and_stash_batch_extras(self, inputs)
        except Exception:
            warn_once(
                self,
                key="batch_extras_failed",
                message=(
                    "Batch-extras extraction failed (best-effort); continuing "
                    "without extra batch diagnostics for this step."
                ),
                exc_info=True,
            )

        cfg = getattr(self, "coord_gaussian_rps_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return super().compute_loss(  # type: ignore[misc]
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if not isinstance(inputs, dict):
            raise TypeError("coord_gaussian_rps enabled but inputs is not a dict")
        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "coord_gaussian_rps enabled but inputs['labels'] is missing or not a torch.Tensor"
            )

        coord_token_ids = self._get_coord_token_ids()
        if not coord_token_ids:
            raise ValueError(
                "coord_gaussian_rps enabled but no coord token ids found; ensure tokenizer provides coord vocab"
            )

        labels_orig = labels
        setattr(self, "_coordexp_labels_for_acc", labels_orig)
        masked_labels = self._mask_coord_targets(labels_orig, coord_token_ids)
        inputs["labels"] = masked_labels

        passed_num_items = num_items_in_batch
        if num_items_in_batch is not None:
            passed_num_items = self._count_supervised_tokens(masked_labels)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=passed_num_items,
        )

        try:
            reporter = SwiftMetricReporter(self)
            self._log_base_ce_metrics(
                reporter=reporter,
                loss_base=loss,
                masked_labels=masked_labels,
            )
            loss = self._maybe_add_coord_gaussian_rps_loss(
                reporter=reporter,
                loss=loss,
                outputs=outputs,
                labels=labels_orig,
                masked_labels=masked_labels,
                extras=extras,
                coord_token_ids=coord_token_ids,
            )

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
    ) -> None:
        reporter.update("base_ce/loss", float(loss_base.detach().cpu().item()))
        noncoord_tokens = int((masked_labels[:, 1:] != -100).sum().detach().item())
        reporter.update("base_ce/noncoord_tokens", float(noncoord_tokens))

    def _maybe_add_coord_gaussian_rps_loss(
        self,
        *,
        reporter: Any,
        loss: torch.Tensor,
        outputs: Any,
        labels: torch.Tensor,
        masked_labels: torch.Tensor,
        extras: Any,
        coord_token_ids: list[int],
    ) -> torch.Tensor:
        cfg = getattr(self, "coord_gaussian_rps_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return loss
        logits = getattr(outputs, "logits", None)
        if logits is None or not isinstance(logits, torch.Tensor):
            raise RuntimeError(
                "coord_gaussian_rps is enabled, but model outputs do not contain logits."
            )

        vocab_size = int(logits.shape[-1])
        coord_id_map = self._get_coord_id_map(vocab_size, logits.device)
        if coord_id_map is None:
            raise RuntimeError(
                "coord_gaussian_rps is enabled, but coord_id_map could not be constructed."
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

        from src.trainers.losses.coord_gaussian_rps import (
            compute_coord_gaussian_rps_loss,
        )

        result = compute_coord_gaussian_rps_loss(
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
        )
        if result is None:
            return loss
        self._log_coord_gaussian_rps_metrics(
            reporter=reporter,
            result=result,
            batch_size=int(labels.shape[0]),
        )
        return loss.float() + result.coord_loss.float()

    def _log_coord_gaussian_rps_metrics(
        self,
        *,
        reporter: Any,
        result: Any,
        batch_size: int,
    ) -> None:
        loss_total = getattr(result, "coord_loss", None)
        if not isinstance(loss_total, torch.Tensor):
            return

        loss_gaussian = getattr(result, "gaussian_contrib", None)
        loss_rps = getattr(result, "rps_contrib", None)
        loss_ce = getattr(result, "ce_contrib", None)
        loss_type_gate = getattr(result, "type_gate_contrib", None)
        coord_tokens = int(getattr(result, "coord_tokens", 0) or 0)
        type_gate_tokens = int(getattr(result, "type_gate_tokens", 0) or 0)

        reporter.update("coord_gaussian_rps/loss", float(loss_total.detach().cpu().item()))
        if isinstance(loss_gaussian, torch.Tensor):
            reporter.update(
                "coord_gaussian_rps/gaussian_ce",
                float(loss_gaussian.detach().cpu().item()),
            )
        if isinstance(loss_rps, torch.Tensor):
            reporter.update("coord_gaussian_rps/rps", float(loss_rps.detach().cpu().item()))
        if isinstance(loss_ce, torch.Tensor):
            reporter.update("coord_gaussian_rps/ce", float(loss_ce.detach().cpu().item()))
        if isinstance(loss_type_gate, torch.Tensor):
            reporter.update(
                "coord_gaussian_rps/type_gate",
                float(loss_type_gate.detach().cpu().item()),
            )
        reporter.update("coord_gaussian_rps/coord_tokens", float(coord_tokens))
        reporter.update("coord_gaussian_rps/type_gate_tokens", float(type_gate_tokens))

        reporter.update("coord_diag/enabled", 1.0)
        reporter.update("coord_diag/loss", float(loss_total.detach().cpu().item()))
        if isinstance(loss_gaussian, torch.Tensor):
            reporter.update(
                "coord_diag/gaussian_ce",
                float(loss_gaussian.detach().cpu().item()),
            )
        if isinstance(loss_rps, torch.Tensor):
            reporter.update("coord_diag/rps", float(loss_rps.detach().cpu().item()))
        if isinstance(loss_ce, torch.Tensor):
            reporter.update("coord_diag/ce", float(loss_ce.detach().cpu().item()))
        if isinstance(loss_type_gate, torch.Tensor):
            reporter.update(
                "coord_diag/type_gate",
                float(loss_type_gate.detach().cpu().item()),
            )
        reporter.update("coord_diag/coord_tokens", float(coord_tokens))

        for key, metric_name in (
            ("type_gate_allowed_mass_mean", "coord_gaussian_rps/type_gate_allowed_mass"),
            ("target_entropy", "coord_gaussian_rps/target_entropy"),
            ("target_peak_prob", "coord_gaussian_rps/target_peak_prob"),
            ("target_r95_radius_mean", "coord_gaussian_rps/target_r95_radius_mean"),
            ("target_r95_radius_max", "coord_gaussian_rps/target_r95_radius_max"),
            ("coord_acc_top5", "coord_diag/acc_top5"),
            ("coord_p_gt_mean", "coord_diag/p_gt_mean"),
            ("coord_margin_mean", "coord_diag/margin_mean"),
            ("coord_expected_bin_mae", "coord_diag/expected_bin_mae"),
            ("coord_expected_bin_abs_err_p90", "coord_diag/expected_bin_abs_err_p90"),
        ):
            value = getattr(result, key, None)
            if isinstance(value, torch.Tensor):
                reporter.update(metric_name, float(value.detach().cpu().item()))

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

    def _count_supervised_tokens(self, labels: torch.Tensor) -> int:
        from src.trainers.losses.coord_gaussian_rps import count_supervised_tokens

        return count_supervised_tokens(labels)

    def _mask_coord_targets(
        self,
        labels: torch.Tensor,
        coord_token_ids: list[int],
    ) -> torch.Tensor:
        from src.trainers.teacher_forcing.stage1 import mask_stage1_coord_targets

        return mask_stage1_coord_targets(labels, coord_token_ids)

    def _get_coord_token_ids(self) -> list[int]:
        cached = getattr(self, "_coord_gaussian_rps_coord_token_ids", None)
        if cached is not None:
            return cached
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        if tokenizer is None:
            return []
        ids = get_coord_token_ids(tokenizer, validate=True)
        setattr(self, "_coord_gaussian_rps_coord_token_ids", ids)
        return ids

    def _get_coord_id_map(
        self,
        vocab_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        cache = getattr(self, "_coord_gaussian_rps_id_map_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_coord_gaussian_rps_id_map_cache", cache)
        key = (str(device), int(vocab_size))
        if key in cache:
            return cache[key]
        coord_token_ids = self._get_coord_token_ids()
        if not coord_token_ids:
            return None

        from src.trainers.losses.coord_gaussian_rps import build_coord_id_map

        id_map = build_coord_id_map(
            vocab_size=int(vocab_size),
            device=device,
            coord_token_ids=coord_token_ids,
        )
        cache[key] = id_map
        return id_map


__all__ = ["CoordGaussianRPSLossMixin"]
