import json
import math
import os
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.coord_tokens.codec import get_coord_token_ids
from src.coord_tokens.loss import topk_expectation_decode
from src.coord_tokens.soft_rasterizer import soft_polygon_mask
from src.data_collators.token_types import TokenType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AggregateTokenTypeMetricsMixin:
    """Trainer mixin to log aggregate loss/accuracy and token-type metrics.

    - Aggregate only (no per-dataset buckets)
    - Safe under packing when token_types are pre-concatenated; skips on mismatch
    - Skips metrics when no supervised tokens to avoid NaNs
    """

    label_field = "dataset_labels"
    segment_field = "dataset_segments"

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        _ = inputs.pop(self.label_field, None)
        _ = inputs.pop(self.segment_field, None)  # Optional legacy field
        labels_for_metrics = inputs.get("labels")
        token_types = inputs.pop("token_types", None)

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            self._log_aggregate_metrics(outputs, labels_for_metrics, token_types)
            self._sync_dataset_metrics()
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def _log_aggregate_metrics(
        self,
        outputs: Any,
        labels: Any,
        token_types: Any,
    ) -> None:
        logits = getattr(outputs, "logits", None)
        if logits is None or labels is None or not isinstance(labels, torch.Tensor):
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

        logits_next = logits[:, :-1, :]
        labels_next = labels[:, 1:]
        mask = labels_next != -100
        supervised_count = int(mask.sum().detach())
        if supervised_count == 0:
            return

        token_types_next = None
        if isinstance(token_types, torch.Tensor) and token_types.shape == labels.shape:
            token_types_next = token_types[:, 1:]
            if not (token_types_next != TokenType.IGNORE).any().item():
                token_types_next = None

        coord_mask = None
        if token_types_next is not None:
            coord_mask = token_types_next == TokenType.COORD
        else:
            coord_mask = self._infer_coord_mask(labels_next, logits_next)

        with torch.no_grad():
            preds = logits_next.argmax(dim=-1)

            labels_masked = labels_next[mask]
            preds_masked = preds[mask]

            topk_indices = None
            flat_logits = logits_next[mask]
            if flat_logits.numel() > 0:
                k = min(5, flat_logits.shape[-1])
                topk_indices = flat_logits.topk(k=k, dim=-1).indices
                top5_acc = (
                    (topk_indices == labels_masked.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                    .mean()
                )
            else:
                top5_acc = logits_next.new_tensor(0.0)

        metrics["token_acc_top5"].update(float(top5_acc.detach().item()))

        if token_types_next is not None:
            types_masked = token_types_next[mask]
            for name, type_id in (
                ("desc", TokenType.DESC),
                ("format", TokenType.FORMAT),
                ("coord", TokenType.COORD),
            ):
                sel = types_masked == type_id
                if not sel.any().item():
                    continue
                metrics[f"{name}_token_frac"].update(float(sel.float().mean().detach().item()))
            for name, type_id in (
                ("desc", TokenType.DESC),
                ("format", TokenType.FORMAT),
            ):
                type_sel = types_masked == type_id
                if not type_sel.any().item():
                    continue
                with torch.no_grad():
                    acc = (
                        (preds_masked[type_sel] == labels_masked[type_sel])
                        .float()
                        .mean()
                    )
                metrics[f"{name}_token_acc"].update(float(acc.detach().item()))

                if topk_indices is None:
                    continue
                with torch.no_grad():
                    top5 = (
                        (topk_indices[type_sel] == labels_masked[type_sel].unsqueeze(-1))
                        .any(dim=-1)
                        .float()
                        .mean()
                    )
                metrics[f"{name}_token_acc_top5"].update(float(top5.detach().item()))

            coord_id_map = None
            try:
                coord_map_fn = getattr(self, "_get_coord_id_map", None)
                if callable(coord_map_fn):
                    coord_id_map = coord_map_fn(
                        logits_next.shape[-1], logits_next.device
                    )
            except Exception:
                coord_id_map = None

            if coord_id_map is not None:
                coord_sel = types_masked == TokenType.COORD
                if coord_sel.any().item():
                    with torch.no_grad():
                        preds_safe = preds_masked.clamp(min=0)
                        pred_bins = coord_id_map[preds_safe]
                        pred_is_coord = pred_bins >= 0

                        coord_pred_is_coord = pred_is_coord[coord_sel].float().mean()
                    metrics["coord_top1_pred_is_coord_rate"].update(
                        float(coord_pred_is_coord.detach().item())
                    )

                    with torch.no_grad():
                        coord_pred_bins = pred_bins[coord_sel]
                        coord_gt_bins = coord_id_map[
                            labels_masked[coord_sel].clamp(min=0)
                        ]
                        valid = coord_pred_bins >= 0
                        if valid.any().item():
                            abs_err = (
                                (coord_pred_bins[valid] - coord_gt_bins[valid])
                                .abs()
                                .float()
                                .mean()
                            )
                            metrics["coord_top1_abs_err_norm"].update(
                                float((abs_err / 1000.0).detach().item())
                            )

                        if topk_indices is not None:
                            top5_hit = (topk_indices == labels_masked.unsqueeze(-1)).any(
                                dim=-1
                            )
                            coord_hit = top5_hit[coord_sel]
                            coord_miss = ~coord_hit
                            if coord_miss.any().item():
                                miss_noncoord = (
                                    (~pred_is_coord[coord_sel])[coord_miss]
                                    .float()
                                    .mean()
                                )
                                metrics["coord_top5_miss_top1_noncoord_rate"].update(
                                    float(miss_noncoord.detach().item())
                                )

        # Token-type metrics (optional)
        if token_types_next is not None:
            text_mask = mask & (token_types_next != TokenType.COORD)
            text_mask = text_mask & (token_types_next != TokenType.IGNORE)
        elif coord_mask is not None:
            text_mask = mask & (~coord_mask)
        else:
            text_mask = mask

        if text_mask is not None and text_mask.any().item():
            with torch.no_grad():
                text_acc = (preds[text_mask] == labels_next[text_mask]).float().mean()
            metrics["text_token_acc"].update(float(text_acc.detach().item()))

        if coord_mask is not None:
            coord_mask = coord_mask & mask
            if coord_mask.any().item():
                with torch.no_grad():
                    coord_logits = logits_next[coord_mask]
                    coord_labels = labels_next[coord_mask]
                    coord_preds = coord_logits.argmax(dim=-1)
                    coord_acc = (coord_preds == coord_labels).float().mean()
                    k = min(5, coord_logits.shape[-1])
                    topk = coord_logits.topk(k=k, dim=-1).indices
                    coord_top5 = (
                        (topk == coord_labels.unsqueeze(-1)).any(dim=-1).float().mean()
                    )
                metrics["coord_token_acc"].update(float(coord_acc.detach().item()))
                metrics["coord_token_acc_top5"].update(
                    float(coord_top5.detach().item())
                )
                coord_id_map = None
                try:
                    coord_map_fn = getattr(self, "_get_coord_id_map", None)
                    if callable(coord_map_fn):
                        coord_id_map = coord_map_fn(
                            logits_next.shape[-1], logits_next.device
                        )
                except Exception:
                    coord_id_map = None
                if coord_id_map is not None and isinstance(topk, torch.Tensor):
                    with torch.no_grad():
                        labels_safe = coord_labels
                        if labels_safe.numel() > 0 and labels_safe.min().item() < 0:
                            labels_safe = labels_safe.clamp(min=0)
                        gt_bins = coord_id_map[labels_safe].to(dtype=torch.long)
                        valid_gt = gt_bins >= 0
                        if valid_gt.any().item():
                            topk_bins = coord_id_map[topk].to(dtype=torch.long)
                            gt_bins = gt_bins[valid_gt]
                            topk_bins = topk_bins[valid_gt]
                            any_coord = (topk_bins >= 0).any(dim=-1)
                            metrics["coord_top5_any_coord_rate"].update(
                                float(any_coord.float().mean().detach().item())
                            )
                            if any_coord.any().item():
                                abs_err = (
                                    topk_bins - gt_bins.unsqueeze(-1)
                                ).abs()
                                abs_err = torch.where(
                                    topk_bins >= 0,
                                    abs_err,
                                    torch.full_like(abs_err, 1_000_000),
                                )
                                min_abs_err = abs_err.min(dim=-1).values
                                min_abs_err = min_abs_err[any_coord].float().mean()
                                metrics["coord_top5_min_abs_err_norm"].update(
                                    float(
                                        (min_abs_err / 1000.0)
                                        .detach()
                                        .cpu()
                                        .item()
                                    )
                                )

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


class CoordAuxLossMixin:
    """Trainer mixin to compute coord auxiliary losses and log metrics."""

    coord_spans_field = "coord_spans"

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        coord_spans = inputs.pop(self.coord_spans_field, None)
        labels_for_metrics = inputs.get("labels")
        loss_scale_for_metrics = inputs.get("loss_scale")
        if isinstance(labels_for_metrics, torch.Tensor):
            try:
                default_keep = True
                if getattr(self, "template", None) is not None:
                    default_keep = bool(
                        getattr(self.template, "sequence_parallel_size", 1) == 1
                    )
                use_logits_to_keep = self.get_use_logits_to_keep(default_keep)
            except Exception:
                use_logits_to_keep = False
            if use_logits_to_keep:
                tmp_inputs = {"labels": labels_for_metrics}
                if loss_scale_for_metrics is not None:
                    tmp_inputs["loss_scale"] = loss_scale_for_metrics
                try:
                    self.prepare_logits_to_keep(tmp_inputs)
                    labels_for_metrics = tmp_inputs.get("labels", labels_for_metrics)
                    loss_scale_for_metrics = tmp_inputs.get(
                        "loss_scale", loss_scale_for_metrics
                    )
                except Exception:
                    pass

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            loss = self._maybe_add_coord_aux_loss(
                loss,
                outputs,
                labels_for_metrics,
                coord_spans,
                num_items_in_batch,
                loss_scale_for_metrics,
            )
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss

    def _maybe_add_coord_aux_loss(
        self,
        loss: torch.Tensor,
        outputs: Any,
        labels: Any,
        coord_spans: Any,
        num_items_in_batch: int | None = None,
        loss_scale: Any = None,
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

        coord_id_map = self._get_coord_id_map(logits_next.shape[-1], logits_next.device)
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
        loss = self._ensure_finite_base_loss(
            loss, outputs, labels_next, num_items_in_batch
        )
        loss = self._maybe_recompute_zero_ce(
            loss,
            logits_next,
            labels_next,
            num_items_in_batch,
            loss_scale,
        )
        if coord_cfg is not None and (
            float(coord_cfg.coord_ce_weight) != 1.0
            or float(coord_cfg.non_coord_ce_weight) != 1.0
        ):
            if loss_scale is None and not getattr(
                self, "_coord_loss_warned_loss_scale", False
            ):
                logger.warning(
                    "coord_loss CE weights set but loss_scale is missing; "
                    "coord/text CE weighting will be skipped"
                )
                self._coord_loss_warned_loss_scale = True

        # NOTE: Base CE in ms-swift can be token-averaged (acc_strategy=token,
        # average_tokens_across_devices=True). To keep aux losses stable across
        # packing lengths and compatible with token averaging, we accumulate
        # per-token sums and normalize with the same denominator where possible.
        l1_sum = logits_next.new_tensor(0.0, dtype=torch.float32)
        bbox_giou_sum = logits_next.new_tensor(0.0, dtype=torch.float32)
        bbox_giou_token_sum = logits_next.new_tensor(0.0, dtype=torch.float32)
        poly_loss_sum = logits_next.new_tensor(0.0, dtype=torch.float32)
        poly_loss_token_sum = logits_next.new_tensor(0.0, dtype=torch.float32)
        poly_iou_sum = logits_next.new_tensor(0.0, dtype=torch.float32)
        poly_smooth_sum = logits_next.new_tensor(0.0, dtype=torch.float32)
        poly_smooth_token_sum = logits_next.new_tensor(0.0, dtype=torch.float32)
        coord_count = 0
        bbox_count = 0
        poly_count = 0
        bbox_token_count = 0
        poly_token_count = 0

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
            ).float()
            target_ids = labels_next[row_idx, coord_positions]
            target_vals = coord_id_map[target_ids].to(dtype=torch.float32) / 1000.0

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
                    pred_poly = pred_slice.view(-1, 2).clamp(0.0, 1.0)
                    target_poly = target_slice.view(-1, 2).clamp(0.0, 1.0)
                    poly_iou = self._poly_mask_iou(
                        pred_poly,
                        target_poly,
                        mask_size=coord_cfg.poly_mask_size,
                        sigma_mask=coord_cfg.poly_sigma_mask,
                        tau_inside=coord_cfg.poly_tau_inside,
                        beta_dist=coord_cfg.poly_beta_dist,
                    )
                    poly_loss_sum = poly_loss_sum + (1.0 - poly_iou)
                    poly_iou_sum = poly_iou_sum + poly_iou
                    smooth = self._poly_smoothness(pred_poly)
                    poly_smooth_sum = poly_smooth_sum + smooth
                    poly_loss_token_sum = poly_loss_token_sum + (1.0 - poly_iou) * float(
                        coord_len
                    )
                    poly_smooth_token_sum = poly_smooth_token_sum + smooth * float(
                        coord_len
                    )
                    poly_token_count += int(coord_len)
                    poly_count += 1
                    continue
                else:
                    continue

                giou = self._giou_loss(pred_box, target_box)
                bbox_giou_sum = bbox_giou_sum + giou
                bbox_giou_token_sum = bbox_giou_token_sum + giou * 4.0
                bbox_token_count += 4
                bbox_count += 1

        if num_items_in_batch is None:
            denom_val = int((labels_next != -100).sum().item())
        elif isinstance(num_items_in_batch, torch.Tensor):
            denom_val = int(num_items_in_batch.item())
        else:
            denom_val = int(num_items_in_batch)
        denom_val = max(denom_val, 0)

        if denom_val <= 0:
            l1_loss = logits_next.new_tensor(0.0)
            giou_loss = logits_next.new_tensor(0.0)
            poly_loss = logits_next.new_tensor(0.0)
            poly_iou = logits_next.new_tensor(0.0)
            poly_smooth = logits_next.new_tensor(0.0)
        else:
            coord_denom = float(coord_count) if coord_count > 0 else 0.0
            bbox_denom = float(bbox_count) if bbox_count > 0 else 0.0
            poly_denom = float(poly_count) if poly_count > 0 else 0.0

            use_token_norm = True
            try:
                acc_strategy = getattr(getattr(self, "args", None), "acc_strategy", None)
                if isinstance(acc_strategy, str) and acc_strategy:
                    use_token_norm = acc_strategy.strip().lower() == "token"
                if bool(
                    getattr(getattr(self, "args", None), "average_tokens_across_devices", False)
                ):
                    use_token_norm = True
            except Exception:
                use_token_norm = True

            if use_token_norm:
                denom = logits_next.new_tensor(float(denom_val), dtype=torch.float32)
                if denom.item() <= 0:
                    denom = logits_next.new_tensor(1.0, dtype=torch.float32)
                # When num_items_in_batch is gathered across devices, HF/ms-swift
                # averages the loss across processes; multiply back to keep the
                # intended global token-average scale.
                scale = logits_next.new_tensor(1.0, dtype=torch.float32)
                if (
                    num_items_in_batch is not None
                    and getattr(self.args, "average_tokens_across_devices", False)
                    and getattr(self, "model_accepts_loss_kwargs", False)
                ):
                    try:
                        scale = logits_next.new_tensor(
                            float(self.accelerator.num_processes), dtype=torch.float32
                        )
                    except Exception:
                        scale = logits_next.new_tensor(1.0, dtype=torch.float32)

                l1_loss = (l1_sum / denom) * scale
                giou_loss = (bbox_giou_token_sum / denom) * scale
                poly_loss = (poly_loss_token_sum / denom) * scale
                poly_smooth = (poly_smooth_token_sum / denom) * scale
            else:
                l1_loss = (
                    l1_sum / coord_denom
                    if coord_denom > 0.0
                    else logits_next.new_tensor(0.0)
                )
                giou_loss = (
                    bbox_giou_sum / bbox_denom
                    if bbox_denom > 0.0
                    else logits_next.new_tensor(0.0)
                )
                poly_loss = (
                    poly_loss_sum / poly_denom
                    if poly_denom > 0.0
                    else logits_next.new_tensor(0.0)
                )
                poly_smooth = (
                    poly_smooth_sum / poly_denom
                    if poly_denom > 0.0
                    else logits_next.new_tensor(0.0)
                )

            poly_iou = (
                poly_iou_sum / poly_denom
                if poly_denom > 0.0
                else logits_next.new_tensor(0.0)
            )

        weighted_l1 = float(coord_cfg.l1_weight) * l1_loss
        weighted_giou = float(coord_cfg.giou_weight) * giou_loss
        poly_mask_weight = getattr(coord_cfg, "poly_mask_weight", coord_cfg.giou_weight)
        weighted_poly_mask = float(poly_mask_weight) * poly_loss
        weighted_poly_smooth = float(coord_cfg.poly_smooth_weight) * poly_smooth

        weighted_l1 = torch.nan_to_num(weighted_l1, nan=0.0, posinf=1e4, neginf=0.0)
        weighted_giou = torch.nan_to_num(weighted_giou, nan=0.0, posinf=1e4, neginf=0.0)
        weighted_poly_mask = torch.nan_to_num(
            weighted_poly_mask, nan=0.0, posinf=1e4, neginf=0.0
        )
        weighted_poly_smooth = torch.nan_to_num(
            weighted_poly_smooth, nan=0.0, posinf=1e4, neginf=0.0
        )

        aux_loss = (
            weighted_l1 + weighted_giou + weighted_poly_mask + weighted_poly_smooth
        )
        loss = loss + aux_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

        coord_ce, desc_ce, loss_ce = self._split_token_ce(
            outputs,
            logits_next,
            labels_next,
            coord_positions_mask,
            num_items_in_batch,
            loss_scale,
        )
        coord_ce = torch.nan_to_num(coord_ce, nan=0.0, posinf=1e4, neginf=0.0)
        desc_ce = torch.nan_to_num(desc_ce, nan=0.0, posinf=1e4, neginf=0.0)
        loss_ce = torch.nan_to_num(loss_ce, nan=0.0, posinf=1e4, neginf=0.0)

        self._log_coord_metrics(
            coord_ce,
            desc_ce,
            loss_ce,
            weighted_l1,
            weighted_giou,
            weighted_poly_mask,
            weighted_poly_smooth,
            poly_iou,
        )
        try:
            self._log_coord_expectation_metrics(
                logits_next=logits_next,
                labels_next=labels_next,
                coord_positions_mask=coord_positions_mask,
            )
        except Exception:
            pass
        return loss

    def _log_coord_metrics(
        self,
        coord_ce: torch.Tensor,
        desc_ce: torch.Tensor,
        loss_ce: torch.Tensor,
        weighted_l1: torch.Tensor,
        weighted_giou: torch.Tensor,
        weighted_poly_mask: torch.Tensor,
        weighted_poly_smooth: torch.Tensor,
        poly_iou: torch.Tensor,
    ) -> None:
        mode = (
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
        )  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        loss_total = loss_ce + weighted_l1 + weighted_giou + weighted_poly_mask + weighted_poly_smooth

        metrics["coord_ce"].update(coord_ce.detach())
        metrics["desc_ce"].update(desc_ce.detach())
        metrics["loss_ce"].update(loss_ce.detach())
        metrics["loss_norm"].update(loss_total.detach())
        if mode == "train":
            metrics["loss"].update(loss_total.detach())

        metrics["coord_loss/l1"].update(weighted_l1.detach())
        metrics["coord_loss/giou"].update(weighted_giou.detach())
        metrics["coord_loss/poly_mask"].update(weighted_poly_mask.detach())
        metrics["coord_loss/poly_smooth"].update(weighted_poly_smooth.detach())

    def _log_coord_expectation_metrics(
        self,
        *,
        logits_next: torch.Tensor,
        labels_next: torch.Tensor,
        coord_positions_mask: torch.Tensor,
    ) -> None:
        cfg = getattr(self, "coord_expectation_metrics_cfg", None)
        if not isinstance(cfg, Mapping) or not bool(cfg.get("enabled", False)):
            return

        coord_token_ids = self._get_coord_token_ids()
        if not coord_token_ids:
            return
        coord_id_map = self._get_coord_id_map(logits_next.shape[-1], logits_next.device)
        if coord_id_map is None:
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

        temperature = float(cfg.get("temperature", 1.0) or 1.0)
        if not math.isfinite(temperature) or temperature <= 0:
            temperature = 1.0

        max_positions = cfg.get("max_positions", 4096)
        try:
            max_positions = int(max_positions)
        except Exception:
            max_positions = 4096
        max_positions = max(0, max_positions)

        tolerances = cfg.get("tolerances", [2])
        if isinstance(tolerances, (int, float)):
            tolerances = [int(tolerances)]
        if not isinstance(tolerances, (list, tuple)):
            tolerances = [0, 1, 2, 4]
        tol_list: list[int] = []
        for t in tolerances:
            try:
                ti = int(t)
            except Exception:
                continue
            if ti < 0:
                continue
            tol_list.append(ti)
        if not tol_list:
            tol_list = [0, 1, 2, 4]
        tol_list = sorted(set(tol_list))
        # Avoid redundant logging: keep at most 2 tolerances (smallest + largest).
        if len(tol_list) > 2:
            tol_list = [tol_list[0], tol_list[-1]]

        mask = coord_positions_mask & (labels_next != -100)
        if not mask.any().item():
            return

        # Flatten coord positions and optionally sub-sample for speed under long packing.
        coord_logits = logits_next[mask]
        coord_labels = labels_next[mask]
        if coord_logits.ndim != 2 or coord_labels.ndim != 1:
            return
        num_pos = int(coord_labels.numel())
        if num_pos <= 0:
            return

        if max_positions > 0 and num_pos > max_positions:
            with torch.no_grad():
                perm = torch.randperm(num_pos, device=coord_labels.device)[:max_positions]
                coord_logits = coord_logits.index_select(0, perm)
                coord_labels = coord_labels.index_select(0, perm)
                num_pos = int(coord_labels.numel())
                if num_pos <= 0:
                    return

        # Map labels (vocab ids) -> coord bin index [0, M-1].
        labels_safe = coord_labels
        if labels_safe.min().item() < 0:
            labels_safe = labels_safe.clamp(min=0)
        target_bins = coord_id_map[labels_safe].to(dtype=torch.long)
        valid_targets = target_bins >= 0
        if not valid_targets.any().item():
            return

        coord_logits = coord_logits[valid_targets]
        target_bins = target_bins[valid_targets]
        if coord_logits.numel() == 0:
            return

        with torch.no_grad():
            idx = torch.tensor(coord_token_ids, device=coord_logits.device, dtype=torch.long)
            coord_logits_sub = coord_logits.index_select(-1, idx)
            coord_logits_sub = coord_logits_sub.float() / float(temperature)
            probs = torch.softmax(coord_logits_sub, dim=-1)

            m = probs.shape[-1]
            # Expected abs error in bins: E[|X - gt|]
            ar = torch.arange(m, device=probs.device, dtype=torch.float32).unsqueeze(0)
            tgt = target_bins.to(dtype=torch.float32).unsqueeze(-1)
            eae = (probs * (ar - tgt).abs()).sum(dim=-1)
            metrics["coord_expect/eae_norm"].update(float((eae.mean() / 1000.0).detach().cpu().item()))

            # Mass within tolerance window around GT: sum_{|i-gt|<=k} p_i
            cdf = probs.cumsum(dim=-1)
            for k in tol_list:
                k = int(k)
                if k < 0:
                    continue
                lo = (target_bins - k).clamp(0, m - 1)
                hi = (target_bins + k).clamp(0, m - 1)
                cdf_hi = cdf.gather(-1, hi.unsqueeze(-1)).squeeze(-1)
                lo_minus = (lo - 1).clamp(0, m - 1)
                cdf_lo = cdf.gather(-1, lo_minus.unsqueeze(-1)).squeeze(-1)
                within = torch.where(lo > 0, cdf_hi - cdf_lo, cdf_hi)
                metrics[f"coord_expect/within_{k}"].update(float(within.mean().detach().cpu().item()))

    def _split_token_ce(
        self,
        outputs: Any,
        logits_next: torch.Tensor,
        labels_next: torch.Tensor,
        coord_positions_mask: torch.Tensor,
        num_items_in_batch: int | None,
        loss_scale: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_per_token = getattr(outputs, "loss", None)
        if loss_per_token is None or not isinstance(loss_per_token, torch.Tensor):
            return self._split_token_ce_from_logits(
                logits_next,
                labels_next,
                coord_positions_mask,
                num_items_in_batch,
                loss_scale,
            )
        loss_per_token = torch.nan_to_num(
            loss_per_token, nan=0.0, posinf=1e4, neginf=0.0
        )

        batch_size = labels_next.shape[0]
        seq_len = labels_next.shape[1]
        expected = batch_size * seq_len
        if loss_per_token.ndim == 0 or loss_per_token.numel() < expected:
            return self._split_token_ce_from_logits(
                logits_next,
                labels_next,
                coord_positions_mask,
                num_items_in_batch,
                loss_scale,
            )
        if loss_per_token.numel() % batch_size != 0:
            return self._split_token_ce_from_logits(
                logits_next,
                labels_next,
                coord_positions_mask,
                num_items_in_batch,
                loss_scale,
            )
        loss_per_token = loss_per_token.reshape(batch_size, -1)
        if loss_per_token.shape[1] < seq_len:
            return self._split_token_ce_from_logits(
                logits_next,
                labels_next,
                coord_positions_mask,
                num_items_in_batch,
                loss_scale,
            )
        loss_next = loss_per_token[:, :seq_len]
        mask = labels_next != -100

        valid_mask = mask
        if valid_mask.any().item() and loss_next[valid_mask].sum().item() == 0.0:
            return self._split_token_ce_from_logits(
                logits_next,
                labels_next,
                coord_positions_mask,
                num_items_in_batch,
                loss_scale,
            )

        coord_mask = coord_positions_mask & valid_mask
        desc_mask = (~coord_positions_mask) & valid_mask

        if num_items_in_batch is None:
            denom = valid_mask.sum()
        else:
            denom = num_items_in_batch
        if isinstance(denom, torch.Tensor):
            denom_t = denom.to(device=loss_next.device, dtype=loss_next.dtype)
        else:
            denom_t = loss_next.new_tensor(float(denom))
        if denom_t.item() <= 0:
            zero = loss_next.new_tensor(0.0)
            return zero, zero, zero

        coord_sum = loss_next[coord_mask].sum() if coord_mask.any().item() else loss_next.new_tensor(0.0)
        desc_sum = loss_next[desc_mask].sum() if desc_mask.any().item() else loss_next.new_tensor(0.0)
        loss_ce = (coord_sum + desc_sum) / denom_t
        coord_ce = coord_sum / denom_t
        desc_ce = desc_sum / denom_t

        return coord_ce, desc_ce, loss_ce

    def _split_token_ce_from_logits(
        self,
        logits_next: torch.Tensor,
        labels_next: torch.Tensor,
        coord_positions_mask: torch.Tensor,
        num_items_in_batch: int | None,
        loss_scale: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits_next is None or labels_next is None:
            zero = torch.tensor(0.0)
            return zero, zero, zero

        vocab_size = logits_next.shape[-1]
        logits_safe = torch.nan_to_num(
            logits_next, nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(min=-1e4, max=1e4)
        ce = F.cross_entropy(
            logits_safe.reshape(-1, vocab_size),
            labels_next.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )

        batch_size = labels_next.shape[0]
        seq_len = labels_next.shape[1]
        ce = ce.reshape(batch_size, seq_len)
        mask = labels_next != -100
        valid_mask = mask
        if isinstance(loss_scale, torch.Tensor):
            loss_scale_next = loss_scale[:, 1 : seq_len + 1]
            if loss_scale_next.numel() == ce.numel():
                loss_scale_next = loss_scale_next.to(device=ce.device, dtype=ce.dtype)
                ce = ce * loss_scale_next

        coord_mask = coord_positions_mask & valid_mask
        desc_mask = (~coord_positions_mask) & valid_mask

        if num_items_in_batch is None:
            denom = valid_mask.sum()
        else:
            denom = num_items_in_batch
        if isinstance(denom, torch.Tensor):
            denom_t = denom.to(device=ce.device, dtype=ce.dtype)
        else:
            denom_t = ce.new_tensor(float(denom))
        if denom_t.item() <= 0:
            zero = ce.new_tensor(0.0)
            return zero, zero, zero

        coord_sum = ce[coord_mask].sum() if coord_mask.any().item() else ce.new_tensor(0.0)
        desc_sum = ce[desc_mask].sum() if desc_mask.any().item() else ce.new_tensor(0.0)
        loss_ce = (coord_sum + desc_sum) / denom_t
        coord_ce = coord_sum / denom_t
        desc_ce = desc_sum / denom_t

        return coord_ce, desc_ce, loss_ce

    def _ensure_finite_base_loss(
        self,
        loss: torch.Tensor,
        outputs: Any,
        labels_next: torch.Tensor,
        num_items_in_batch: int | None,
    ) -> torch.Tensor:
        if torch.isfinite(loss).all().item():
            return loss

        loss_per_token = getattr(outputs, "loss", None)
        if isinstance(loss_per_token, torch.Tensor) and loss_per_token.numel() > 0:
            loss_per_token = torch.nan_to_num(
                loss_per_token, nan=0.0, posinf=1e4, neginf=0.0
            )
            batch_size = labels_next.shape[0]
            seq_len = labels_next.shape[1]
            if loss_per_token.numel() % batch_size != 0:
                return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
            loss_per_token = loss_per_token.reshape(batch_size, -1)[:, :seq_len]
            if num_items_in_batch is None:
                denom = (labels_next != -100).sum()
            else:
                denom = num_items_in_batch
            if isinstance(denom, torch.Tensor):
                denom_t = denom.to(
                    dtype=loss_per_token.dtype, device=loss_per_token.device
                )
            else:
                denom_t = loss_per_token.new_tensor(float(denom))
            if denom_t.item() > 0:
                self._warn_non_finite_loss_once("base_loss")
                return loss_per_token.sum() / denom_t

        self._warn_non_finite_loss_once("base_loss_fallback")
        return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

    def _maybe_recompute_zero_ce(
        self,
        loss: torch.Tensor,
        logits_next: torch.Tensor,
        labels_next: torch.Tensor,
        num_items_in_batch: int | None,
        loss_scale: Any,
    ) -> torch.Tensor:
        if not torch.isfinite(loss).all().item():
            return loss
        if loss.detach().abs().item() > 0:
            return loss

        supervised_mask = labels_next != -100
        if not supervised_mask.any().item():
            return loss

        coord_cfg = getattr(self, "coord_loss_cfg", None)
        if coord_cfg is not None:
            if (
                float(coord_cfg.coord_ce_weight) == 0.0
                and float(coord_cfg.non_coord_ce_weight) == 0.0
            ):
                return loss

        vocab_size = logits_next.shape[-1]
        logits_safe = torch.nan_to_num(
            logits_next, nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(min=-1e4, max=1e4)
        ce = F.cross_entropy(
            logits_safe.reshape(-1, vocab_size),
            labels_next.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )

        if isinstance(loss_scale, torch.Tensor):
            seq_len = labels_next.shape[1]
            loss_scale_next = loss_scale[:, 1 : seq_len + 1]
            if loss_scale_next.numel() == ce.numel():
                ce = ce * loss_scale_next.reshape(-1)

        if num_items_in_batch is None:
            denom = supervised_mask.sum()
        else:
            denom = num_items_in_batch
        if isinstance(denom, torch.Tensor):
            denom_t = denom.to(dtype=ce.dtype, device=ce.device)
        else:
            denom_t = ce.new_tensor(float(denom))
        if denom_t.item() <= 0:
            return loss

        ce_loss = ce.sum() / denom_t
        if (
            num_items_in_batch is not None
            and getattr(self.args, "average_tokens_across_devices", False)
            and getattr(self, "model_accepts_loss_kwargs", False)
        ):
            ce_loss = ce_loss * float(self.accelerator.num_processes)

        self._warn_non_finite_loss_once("base_loss_zero")
        return ce_loss

    def _warn_non_finite_loss_once(self, key: str) -> None:
        flags = getattr(self, "_coord_loss_warn_flags", None)
        if flags is None:
            flags = set()
            setattr(self, "_coord_loss_warn_flags", flags)
        if key in flags:
            return
        flags.add(key)
        logger.warning(
            "Non-finite CE loss detected; falling back to a sanitized loss. "
            "Check logits/labels for NaN/inf."
        )

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
        id_map = torch.full((int(vocab_size),), -1, dtype=torch.long, device=device)
        coord_ids = torch.tensor(coord_token_ids, device=device, dtype=torch.long)
        values = torch.arange(coord_ids.numel(), device=device, dtype=torch.long)
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
    def _poly_mask_iou(
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        mask_size: int,
        sigma_mask: float,
        tau_inside: float,
        beta_dist: float,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        # Force numerically sensitive polygon rasterization to run in float32.
        # Even if inputs are float32, AMP autocast can still downcast some ops,
        # which increases the chance of NaN/Inf gradients (esp. atan2 / logsumexp).
        try:
            autocast_ctx = torch.autocast
            device_type = pred.device.type
            ctx = autocast_ctx(device_type=device_type, enabled=False)
        except Exception:
            ctx = torch.cuda.amp.autocast(enabled=False)  # type: ignore[attr-defined]

        with ctx:
            # Use a conservative epsilon for IoU division; too-small eps can create
            # huge gradients when masks are tiny/empty early in training.
            eps_iou = float(eps)
            if eps_iou < 1e-4:
                eps_iou = 1e-4

            pred_f = (
                torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
                .clamp(0.0, 1.0)
                .float()
            )
            target_f = (
                torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
                .clamp(0.0, 1.0)
                .float()
            )
            pred_mask = soft_polygon_mask(
                pred_f,
                mask_size=mask_size,
                sigma_mask=sigma_mask,
                tau_inside=tau_inside,
                beta_dist=beta_dist,
                eps=eps,
            )
            target_mask = soft_polygon_mask(
                target_f,
                mask_size=mask_size,
                sigma_mask=sigma_mask,
                tau_inside=tau_inside,
                beta_dist=beta_dist,
                eps=eps,
            )
            pred_mask = torch.nan_to_num(pred_mask, nan=0.0, posinf=1.0, neginf=0.0)
            target_mask = torch.nan_to_num(target_mask, nan=0.0, posinf=1.0, neginf=0.0)
            inter = (pred_mask * target_mask).sum()
            union = (pred_mask + target_mask - pred_mask * target_mask).sum()
            iou = inter / (union + eps_iou)
            # If the polygon rasterizer produces NaNs/Infs, treat it as "skip" by
            # returning IoU=1 (poly loss 0) so the step won't explode.
            iou = torch.nan_to_num(iou, nan=1.0, posinf=1.0, neginf=0.0)
            return iou.clamp(0.0, 1.0)

    @staticmethod
    def _poly_smoothness(vertices: torch.Tensor) -> torch.Tensor:
        vertices = torch.nan_to_num(vertices, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0.0, 1.0
        )
        prev = torch.roll(vertices, shifts=1, dims=0)
        nxt = torch.roll(vertices, shifts=-1, dims=0)
        smooth = nxt - 2.0 * vertices + prev
        return (smooth * smooth).sum(dim=-1).mean()

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


class InstabilityMonitorMixin:
    """Per-step monitor + optional guard for catastrophic batches.

    Enabled via `custom.instability_monitor` (stored under `CustomConfig.extra`).
    The collator attaches `instability_meta_json` (batch sample_id/base_idx info),
    which this mixin consumes before model forward.

    When triggered, writes a JSON event to `<output_dir>/instability_dumps/events.jsonl`
    (or `dump_dir` if provided). If guard is enabled, replaces the loss with 0.0 to
    avoid poisoning weights for the current step.
    """

    meta_field = "instability_meta_json"

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch=None
    ):
        meta_json = inputs.pop(self.meta_field, None)
        labels_snapshot = inputs.get("labels")

        loss, outputs = super().compute_loss(  # type: ignore[misc]
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        try:
            loss = self._monitor_and_guard(
                loss=loss, outputs=outputs, labels=labels_snapshot, meta_json=meta_json
            )
        except Exception:
            # Best-effort only; never block training.
            pass

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return float(out)

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _get_cfg(self) -> Mapping[str, Any]:
        raw = getattr(self, "instability_monitor_cfg", None)
        return raw if isinstance(raw, Mapping) else {}

    def _rank(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        try:
            return int(dist.get_rank())
        except Exception:
            return 0

    def _is_main_process(self) -> bool:
        return self._rank() == 0

    def _dump_dir(self, cfg: Mapping[str, Any]) -> str | None:
        raw = cfg.get("dump_dir")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        output_dir = getattr(getattr(self, "args", None), "output_dir", None)
        if isinstance(output_dir, str) and output_dir:
            return os.path.join(output_dir, "instability_dumps")
        return None

    def _append_event(self, dump_dir: str, event: Mapping[str, Any]) -> None:
        if not self._is_main_process():
            return
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, "events.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(event), ensure_ascii=True, sort_keys=True))
            f.write("\n")

    @staticmethod
    def _load_jsonl_records_by_index(
        jsonl_path: str, indices: Sequence[int]
    ) -> dict[int, Any]:
        """Load a subset of JSONL rows by 0-based line index (best-effort)."""
        wanted = sorted({int(i) for i in indices if isinstance(i, int) or str(i).isdigit()})
        if not wanted:
            return {}
        out: dict[int, Any] = {}
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                current = 0
                target_pos = 0
                target = wanted[target_pos]
                for line in f:
                    if current == target:
                        raw = line.strip("\n")
                        try:
                            out[current] = json.loads(raw)
                        except Exception:
                            out[current] = {"_raw": raw, "_parse_error": True}
                        target_pos += 1
                        if target_pos >= len(wanted):
                            break
                        target = wanted[target_pos]
                    current += 1
        except Exception as exc:
            return {"_error": str(exc)}  # type: ignore[return-value]
        return out

    def _maybe_dump_samples(
        self, dump_dir: str, *, mode: str, step: Any, meta_json: str | None
    ) -> None:
        cfg = self._get_cfg()
        if not bool(cfg.get("dump_samples", False)):
            return
        if not self._is_main_process():
            return
        if not isinstance(meta_json, str) or not meta_json.strip():
            return

        jsonl_path = None
        if mode == "train":
            jp = getattr(self, "instability_train_jsonl", None)
            if isinstance(jp, str) and jp:
                jsonl_path = jp
        else:
            jp = getattr(self, "instability_val_jsonl", None)
            if isinstance(jp, str) and jp:
                jsonl_path = jp
        if jsonl_path is None:
            return

        try:
            meta = json.loads(meta_json)
        except Exception:
            meta = None
        if not isinstance(meta, list):
            return

        base_idxs: list[int] = []
        # meta is a list of packs; each contains {"samples":[{"base_idx":...}, ...]}
        for pack in meta:
            if not isinstance(pack, dict):
                continue
            samples = pack.get("samples")
            if not isinstance(samples, list):
                continue
            for s in samples:
                if not isinstance(s, dict):
                    continue
                bi = s.get("base_idx")
                try:
                    bi_i = int(bi)
                except Exception:
                    continue
                if bi_i >= 0:
                    base_idxs.append(bi_i)

        if not base_idxs:
            return

        records = self._load_jsonl_records_by_index(jsonl_path, base_idxs)
        payload = {
            "mode": mode,
            "global_step": step,
            "jsonl_path": jsonl_path,
            "base_idxs": sorted(set(base_idxs)),
            "records": records,
            "meta": meta,
        }

        os.makedirs(dump_dir, exist_ok=True)
        out_path = os.path.join(dump_dir, f"samples_step{step}_{mode}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, sort_keys=True)

    def _ema_update(self, name: str, value: float, decay: float) -> float:
        key = f"_instab_ema_{name}"
        prev = getattr(self, key, None)
        ema = float(value) if prev is None else float(decay) * float(prev) + (1.0 - float(decay)) * float(value)
        setattr(self, key, ema)
        return ema

    def _request_training_stop(self, *, reason: str, mode: str, step: Any) -> None:
        """Best-effort early stop request that works with HF Trainer-style loops."""
        if getattr(self, "_instab_stop_requested", False):
            return
        setattr(self, "_instab_stop_requested", True)

        # Mark the control/state so Trainer breaks out cleanly at the next check.
        try:
            control = getattr(self, "control", None)
            if control is not None:
                try:
                    control.should_training_stop = True
                except Exception:
                    pass
                try:
                    control.should_epoch_stop = True
                except Exception:
                    pass
        except Exception:
            pass
        try:
            state = getattr(self, "state", None)
            if state is not None:
                setattr(state, "should_training_stop", True)
        except Exception:
            pass

        if self._is_main_process():
            logger.error(
                "InstabilityMonitor early stop requested (mode=%s step=%s reason=%s)",
                mode,
                step,
                reason,
            )

    def _monitor_and_guard(
        self, *, loss: torch.Tensor, outputs: Any, labels: Any, meta_json: Any
    ) -> torch.Tensor:
        cfg = self._get_cfg()
        if not bool(cfg.get("enabled", False)):
            return loss

        guard_cfg = cfg.get("guard", {}) if isinstance(cfg.get("guard"), Mapping) else {}
        guard_enabled = bool(guard_cfg.get("enabled", False))

        # Optional early-stop (abort training cleanly) when a catastrophic signal is detected.
        early_cfg = cfg.get("early_stop", {}) if isinstance(cfg.get("early_stop"), Mapping) else {}
        early_enabled = bool(early_cfg.get("enabled", False))

        mode = (
            "train"
            if getattr(self, "model", None) is None or self.model.training
            else "eval"
        )  # type: ignore[attr-defined]
        if mode != "train" and not bool(guard_cfg.get("guard_in_eval", False)):
            guard_enabled = False

        ema_decay = self._as_float(cfg.get("ema_decay", 0.98), 0.98)
        ema_decay = min(max(ema_decay, 0.0), 0.9999)
        spike_factor = self._as_float(cfg.get("spike_factor", 8.0), 8.0)
        abs_loss_threshold = self._as_float(cfg.get("abs_loss_threshold", 0.2), 0.2)
        min_supervised_tokens = self._as_int(cfg.get("min_supervised_tokens", 64), 64)
        min_token_acc = self._as_float(guard_cfg.get("min_token_acc", 0.01), 0.01)
        max_abs_logit = self._as_float(guard_cfg.get("max_abs_logit", 200.0), 200.0)

        loss_val = float(loss.detach().float().cpu().item())
        finite_loss = bool(torch.isfinite(loss).all().item())

        logits = getattr(outputs, "logits", None)
        token_acc = None
        supervised_tokens = None
        max_logit_abs = None
        finite_logits = None

        if isinstance(logits, torch.Tensor) and isinstance(labels, torch.Tensor):
            seq_len = min(int(logits.shape[1]), max(int(labels.shape[1] - 1), 0))
            if seq_len > 0:
                logits_next = logits[:, :seq_len, :]
                labels_next = labels[:, 1 : seq_len + 1]
                mask = labels_next != -100
                supervised_tokens = int(mask.sum().detach().cpu().item())
                with torch.no_grad():
                    if supervised_tokens > 0:
                        logits_sup = logits_next[mask]
                        labels_sup = labels_next[mask]
                        finite_logits = bool(torch.isfinite(logits_sup).all().item())
                        safe_logits = torch.nan_to_num(
                            logits_sup, nan=0.0, posinf=1e4, neginf=-1e4
                        )
                        max_logit_abs = float(
                            safe_logits.abs().amax().detach().cpu().item()
                        )
                        preds = safe_logits.argmax(dim=-1)
                        token_acc = float(
                            (preds == labels_sup).float().mean().detach().cpu().item()
                        )
                    else:
                        # No supervised tokens -> ignore logits checks for this batch.
                        finite_logits = True

        ema_loss = None
        if mode == "train":
            ema_loss = self._ema_update("loss", loss_val, ema_decay)

        reasons: list[str] = []
        if not finite_loss and bool(guard_cfg.get("guard_on_nonfinite", True)):
            reasons.append("nonfinite_loss")
        if finite_logits is False and bool(guard_cfg.get("guard_on_nonfinite", True)):
            reasons.append("nonfinite_logits")
        if max_logit_abs is not None and max_logit_abs > max_abs_logit:
            reasons.append("logit_overflow_like")

        is_spike = False
        if mode == "train" and ema_loss is not None:
            if loss_val >= abs_loss_threshold and ema_loss > 0 and loss_val > float(spike_factor) * float(ema_loss):
                is_spike = True
            if loss_val >= abs_loss_threshold and ema_loss < abs_loss_threshold / 4:
                is_spike = True
        if is_spike and bool(guard_cfg.get("guard_on_spike", True)):
            reasons.append("loss_spike")

        if (
            token_acc is not None
            and supervised_tokens is not None
            and supervised_tokens >= min_supervised_tokens
            and token_acc <= 0.0
            and bool(guard_cfg.get("guard_on_zero_acc", True))
        ):
            reasons.append("zero_token_acc")
        elif (
            token_acc is not None
            and supervised_tokens is not None
            and supervised_tokens >= min_supervised_tokens
            and token_acc < min_token_acc
            and bool(guard_cfg.get("guard_on_zero_acc", True))
            and is_spike
        ):
            reasons.append("low_token_acc")

        if not reasons:
            return loss

        dump_dir = self._dump_dir(cfg)
        step = getattr(getattr(self, "state", None), "global_step", None)
        epoch = getattr(getattr(self, "state", None), "epoch", None)

        # Decide whether to request an early stop. Default is off, but when enabled we
        # stop on zero token acc (strong crash indicator) to save time.
        should_early_stop = False
        stop_reason = None
        if mode == "train":
            if bool(cfg.get("early_stop_on_zero_acc", False)) and "zero_token_acc" in reasons:
                should_early_stop = True
                stop_reason = "zero_token_acc"
            if early_enabled:
                raw_on = early_cfg.get("on_reasons")
                if raw_on is None:
                    on_reasons = ("zero_token_acc",)
                elif isinstance(raw_on, str):
                    on_reasons = (raw_on,)
                elif isinstance(raw_on, (list, tuple, set)):
                    on_reasons = tuple(str(r) for r in raw_on)
                else:
                    on_reasons = ("zero_token_acc",)
                for r in reasons:
                    if r in on_reasons:
                        should_early_stop = True
                        stop_reason = r
                        break
        lr = None
        try:
            opt = getattr(self, "optimizer", None)
            if opt is not None and getattr(opt, "param_groups", None):
                lr = opt.param_groups[0].get("lr", None)
        except Exception:
            lr = None
        if lr is None:
            try:
                sched = getattr(self, "lr_scheduler", None)
                if sched is not None:
                    last = sched.get_last_lr()
                    if last:
                        lr = last[0]
            except Exception:
                lr = None
        if lr is None:
            lr = getattr(getattr(self, "args", None), "learning_rate", None)

        meta_str = meta_json if isinstance(meta_json, str) else None
        if isinstance(meta_str, str) and len(meta_str) > 20000:
            meta_str = meta_str[:20000] + "...(truncated)"

        event = {
            "mode": mode,
            "reasons": reasons,
            "global_step": step,
            "epoch": epoch,
            "loss": loss_val,
            "ema_loss": ema_loss,
            "learning_rate": lr,
            "supervised_tokens": supervised_tokens,
            "token_acc": token_acc,
            "max_abs_logit": max_logit_abs,
            "finite_loss": finite_loss,
            "finite_logits": finite_logits,
            "meta_json": meta_str,
        }
        if dump_dir is not None:
            try:
                self._append_event(dump_dir, event)
                self._maybe_dump_samples(
                    dump_dir,
                    mode=mode,
                    step=step,
                    meta_json=meta_str,
                )
            except Exception:
                pass

        if self._is_main_process():
            ema_s = None if ema_loss is None else f"{float(ema_loss):.6f}"
            acc_s = None if token_acc is None else f"{float(token_acc):.4f}"
            maxlog_s = (
                None if max_logit_abs is None else f"{float(max_logit_abs):.2f}"
            )
            logger.warning(
                "InstabilityMonitor triggered (mode=%s step=%s): %s (loss=%.6f ema=%s acc=%s sup=%s max|logit|=%s)",
                mode,
                step,
                ",".join(reasons),
                loss_val,
                ema_s,
                acc_s,
                supervised_tokens,
                maxlog_s,
            )

        if should_early_stop and stop_reason is not None:
            # Ensure we don't backprop through a bad step; stop ASAP after dumping.
            guard_enabled = True
            try:
                self._request_training_stop(reason=stop_reason, mode=mode, step=step)
            except Exception:
                pass

        if not guard_enabled:
            return loss

        # IMPORTANT: return a differentiable zero. A detached 0.0 breaks
        # DeepSpeed/Accelerate backward with:
        #   "element 0 of tensors does not require grad and does not have a grad_fn"
        #
        # Also avoid `loss * 0.0` when loss is NaN/Inf (NaN*0 -> NaN).
        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            return safe_logits.sum() * 0.0
        return loss * 0.0
