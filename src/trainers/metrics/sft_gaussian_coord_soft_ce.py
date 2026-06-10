from __future__ import annotations

from typing import Any

import torch

from src.coord_tokens.codec import get_coord_token_ids


class SFTGaussianCoordSoftCELossMixin:
    """Add packed-SFT Gaussian coord soft CE without masking base SFT labels."""

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        from src.metrics.reporter import SwiftMetricReporter

        cfg = getattr(self, "sft_gaussian_coord_soft_ce_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return super().compute_loss(  # type: ignore[misc]
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        if not isinstance(inputs, dict):
            raise TypeError("sft gaussian coord soft CE enabled but inputs is not a dict")
        labels = inputs.get("labels")
        if labels is None or not isinstance(labels, torch.Tensor):
            raise ValueError(
                "sft gaussian coord soft CE enabled but inputs['labels'] is missing"
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
                "sft gaussian coord soft CE is enabled, but model outputs do not contain logits"
            )

        coord_token_ids = self._get_sft_gaussian_coord_token_ids()
        if not coord_token_ids:
            raise RuntimeError(
                "sft gaussian coord soft CE enabled but tokenizer has no coord tokens"
            )
        coord_id_map = self._get_sft_gaussian_coord_id_map(
            vocab_size=int(logits.shape[-1]),
            device=logits.device,
            coord_token_ids=coord_token_ids,
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

        from src.trainers.losses.sft_gaussian_coord_soft_ce import (
            compute_sft_gaussian_coord_soft_ce_loss,
        )

        result = compute_sft_gaussian_coord_soft_ce_loss(
            logits=logits,
            labels=labels,
            coord_token_ids=coord_token_ids,
            coord_id_map=coord_id_map,
            cfg=cfg,
            average_tokens_across_devices=avg_tokens,
            model_accepts_loss_kwargs=model_accepts,
            accelerator_num_processes=acc_num_proc,
        )
        if result is None:
            return (loss, outputs) if return_outputs else loss

        reporter = SwiftMetricReporter(self)
        self._log_sft_gaussian_coord_soft_ce_metrics(reporter=reporter, result=result)
        loss = loss + result.loss
        return (loss, outputs) if return_outputs else loss

    def _get_sft_gaussian_coord_token_ids(self) -> list[int]:
        cached = getattr(self, "_sft_gaussian_coord_token_ids", None)
        if cached is not None:
            return cached
        tokenizer = getattr(getattr(self, "template", None), "tokenizer", None)
        if tokenizer is None:
            return []
        ids = get_coord_token_ids(tokenizer, validate=True)
        setattr(self, "_sft_gaussian_coord_token_ids", ids)
        return ids

    def _get_sft_gaussian_coord_id_map(
        self,
        *,
        vocab_size: int,
        device: torch.device,
        coord_token_ids: list[int],
    ) -> torch.Tensor:
        cache = getattr(self, "_sft_gaussian_coord_id_map_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_sft_gaussian_coord_id_map_cache", cache)
        key = (str(device), int(vocab_size))
        if key in cache:
            return cache[key]

        from src.trainers.losses.coord_soft_ce_w1 import build_coord_id_map

        id_map = build_coord_id_map(
            vocab_size=int(vocab_size),
            device=device,
            coord_token_ids=coord_token_ids,
        )
        cache[key] = id_map
        return id_map

    def _log_sft_gaussian_coord_soft_ce_metrics(
        self,
        *,
        reporter: Any,
        result: Any,
    ) -> None:
        def _scalar(value: Any) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)

        reporter.update_many(
            {
                "sft_coord_soft_ce/enabled": 1.0,
                "sft_coord_soft_ce/loss": _scalar(result.loss),
                "sft_coord_soft_ce/coord_tokens": float(int(result.coord_tokens)),
                "sft_coord_soft_ce/target_entropy": _scalar(result.target_entropy),
                "sft_coord_soft_ce/target_peak_prob": _scalar(result.target_peak_prob),
                "sft_coord_soft_ce/target_r95_radius_mean": _scalar(
                    result.target_r95_radius_mean
                ),
                "sft_coord_soft_ce/target_r95_radius_max": _scalar(
                    result.target_r95_radius_max
                ),
                "sft_coord_soft_ce/coord_acc_top5": _scalar(result.coord_acc_top5),
                "sft_coord_soft_ce/coord_p_gt": _scalar(result.coord_p_gt_mean),
                "sft_coord_soft_ce/expected_bin_mae": _scalar(
                    result.expected_bin_mae
                ),
                "sft_coord_soft_ce/no_trie_or_type_gate": 1.0,
            }
        )


__all__ = ["SFTGaussianCoordSoftCELossMixin"]
