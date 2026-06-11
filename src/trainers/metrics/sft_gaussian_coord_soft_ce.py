from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from src.coord_tokens.codec import get_coord_token_ids


class SFTGaussianCoordSoftCELossMixin:
    """Add packed-SFT Gaussian coord soft CE without masking base SFT labels."""

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples, num_items = super().get_batch_samples(  # type: ignore[misc]
            epoch_iterator,
            num_batches,
            device,
        )
        cfg = getattr(self, "sft_gaussian_coord_soft_ce_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            setattr(self, "_sft_gaussian_coord_num_items_in_batch", None)
            return batch_samples, num_items

        denom = self._count_sft_gaussian_coord_tokens_in_samples(
            batch_samples,
            device=device,
        )
        avg_tokens = bool(
            getattr(getattr(self, "args", None), "average_tokens_across_devices", False)
        )
        model_accepts = bool(getattr(self, "model_accepts_loss_kwargs", False))
        if (
            avg_tokens
            and model_accepts
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(denom, op=dist.ReduceOp.SUM)
        setattr(self, "_sft_gaussian_coord_num_items_in_batch", denom.detach())
        return batch_samples, num_items

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
        aux_loss = result.loss
        if getattr(self, "model", None) is not None and bool(self.model.training):
            if (
                bool(getattr(self, "model_accepts_loss_kwargs", False))
                and num_items_in_batch is not None
                and getattr(self, "compute_loss_func", None) is None
            ):
                window_denom = getattr(
                    self,
                    "_sft_gaussian_coord_num_items_in_batch",
                    None,
                )
                if isinstance(window_denom, torch.Tensor):
                    window_denom = window_denom.to(
                        device=result.loss_sum.device,
                        dtype=result.loss_sum.dtype,
                    )
                    if bool((window_denom > 0).detach().item()):
                        aux_loss = result.loss_sum / window_denom
                        if avg_tokens and model_accepts:
                            if dist.is_available() and dist.is_initialized():
                                scale = float(dist.get_world_size())
                            else:
                                scale = float(acc_num_proc or 1)
                            aux_loss = aux_loss * scale
                        aux_loss = torch.nan_to_num(
                            aux_loss,
                            nan=0.0,
                            posinf=1e4,
                            neginf=0.0,
                        )
        loss = loss + aux_loss
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

    def _count_sft_gaussian_coord_tokens_in_samples(
        self,
        batch_samples: Any,
        *,
        device: torch.device | str,
    ) -> torch.Tensor:
        coord_token_ids = self._get_sft_gaussian_coord_token_ids()
        target_device = torch.device(device)
        denom = torch.zeros((), dtype=torch.float32, device=target_device)
        if not coord_token_ids:
            return denom

        coord_ids_by_device: dict[torch.device, torch.Tensor] = {}
        for sample in batch_samples or ():
            if not isinstance(sample, dict):
                continue
            labels = sample.get("labels")
            if not isinstance(labels, torch.Tensor) or labels.ndim == 0:
                continue
            if int(labels.shape[-1]) <= 1:
                continue
            labels_next = labels[..., 1:]
            label_device = labels_next.device
            coord_ids = coord_ids_by_device.get(label_device)
            if coord_ids is None:
                coord_ids = torch.tensor(
                    coord_token_ids,
                    dtype=torch.long,
                    device=label_device,
                )
                coord_ids_by_device[label_device] = coord_ids
            coord_mask = (labels_next != -100) & torch.isin(labels_next, coord_ids)
            denom = denom + coord_mask.sum().to(
                device=target_device,
                dtype=torch.float32,
            )
        return denom

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
