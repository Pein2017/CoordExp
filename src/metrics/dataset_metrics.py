from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.data_collators.token_types import TokenType
from src.coord_tokens.codec import get_coord_token_ids
from src.coord_tokens.loss import topk_expectation_decode
from src.coord_tokens.soft_rasterizer import soft_polygon_mask
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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
            seg_acc = (preds[mask] == labels_next[mask]).float().mean()

            flat_logits = logits_next[mask]
            if flat_logits.numel() > 0:
                k = min(5, flat_logits.shape[-1])
                topk = flat_logits.topk(k=k, dim=-1).indices
                labels_flat = labels_next[mask].unsqueeze(-1)
                top5_acc = (topk == labels_flat).any(dim=-1).float().mean()
            else:
                top5_acc = seg_acc

        metrics["token_acc"].update(float(seg_acc.detach().item()))
        metrics["token_acc_top5"].update(float(top5_acc.detach().item()))

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
                    coord_top5 = (topk == coord_labels.unsqueeze(-1)).any(dim=-1).float().mean()
                metrics["coord_token_acc"].update(float(coord_acc.detach().item()))
                metrics["coord_token_acc_top5"].update(float(coord_top5.detach().item()))

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
        num_items_in_batch: int | None,
        loss_scale: Any,
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

        l1_sum = logits_next.new_tensor(0.0)
        bbox_giou_sum = logits_next.new_tensor(0.0)
        poly_loss_sum = logits_next.new_tensor(0.0)
        poly_smooth_sum = logits_next.new_tensor(0.0)
        coord_count = 0
        bbox_count = 0
        poly_count = 0

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
                    poly_smooth_sum = poly_smooth_sum + self._poly_smoothness(
                        pred_poly
                    )
                    poly_count += 1
                    continue
                else:
                    continue

                bbox_giou_sum = bbox_giou_sum + self._giou_loss(pred_box, target_box)
                bbox_count += 1

        l1_loss = (
            l1_sum / float(coord_count)
            if coord_count > 0
            else logits_next.new_tensor(0.0)
        )
        giou_loss = (
            bbox_giou_sum / float(bbox_count)
            if bbox_count > 0
            else logits_next.new_tensor(0.0)
        )
        poly_loss = (
            poly_loss_sum / float(poly_count)
            if poly_count > 0
            else logits_next.new_tensor(0.0)
        )
        poly_smooth = (
            poly_smooth_sum / float(poly_count)
            if poly_count > 0
            else logits_next.new_tensor(0.0)
        )

        weighted_l1 = float(coord_cfg.l1_weight) * l1_loss
        weighted_giou = float(coord_cfg.giou_weight) * giou_loss
        weighted_poly_mask = float(coord_cfg.giou_weight) * poly_loss
        weighted_poly_smooth = float(coord_cfg.poly_smooth_weight) * poly_smooth

        weighted_l1 = torch.nan_to_num(
            weighted_l1, nan=0.0, posinf=1e4, neginf=0.0
        )
        weighted_giou = torch.nan_to_num(
            weighted_giou, nan=0.0, posinf=1e4, neginf=0.0
        )
        weighted_poly_mask = torch.nan_to_num(
            weighted_poly_mask, nan=0.0, posinf=1e4, neginf=0.0
        )
        weighted_poly_smooth = torch.nan_to_num(
            weighted_poly_smooth, nan=0.0, posinf=1e4, neginf=0.0
        )

        aux_loss = weighted_l1 + weighted_giou + weighted_poly_mask + weighted_poly_smooth
        loss = loss + aux_loss
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

        coord_ce, desc_ce = self._split_token_ce(
            outputs,
            logits_next,
            labels_next,
            coord_positions_mask,
            num_items_in_batch,
            loss_scale,
        )
        coord_ce = torch.nan_to_num(coord_ce, nan=0.0, posinf=1e4, neginf=0.0)
        desc_ce = torch.nan_to_num(desc_ce, nan=0.0, posinf=1e4, neginf=0.0)

        self._log_coord_metrics(
            coord_ce,
            desc_ce,
            weighted_l1,
            weighted_giou,
            weighted_poly_mask,
            weighted_poly_smooth,
        )
        return loss

    def _log_coord_metrics(
        self,
        coord_ce: torch.Tensor,
        desc_ce: torch.Tensor,
        weighted_l1: torch.Tensor,
        weighted_giou: torch.Tensor,
        weighted_poly_mask: torch.Tensor,
        weighted_poly_smooth: torch.Tensor,
    ) -> None:
        mode = "train" if getattr(self, "model", None) is None or self.model.training else "eval"  # type: ignore[attr-defined]
        custom_metrics = getattr(self, "custom_metrics", None)
        if custom_metrics is None or mode not in custom_metrics:
            return
        metrics = custom_metrics[mode]

        metrics["coord_ce"].update(coord_ce.detach())
        metrics["desc_ce"].update(desc_ce.detach())
        metrics["l1"].update(weighted_l1.detach())
        metrics["giou"].update(weighted_giou.detach())
        metrics["poly_mask"].update(weighted_poly_mask.detach())
        metrics["poly_smooth"].update(weighted_poly_smooth.detach())

    def _split_token_ce(
        self,
        outputs: Any,
        logits_next: torch.Tensor,
        labels_next: torch.Tensor,
        coord_positions_mask: torch.Tensor,
        num_items_in_batch: int | None,
        loss_scale: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if loss_per_token.numel() % batch_size != 0:
            return self._split_token_ce_from_logits(
                logits_next,
                labels_next,
                coord_positions_mask,
                num_items_in_batch,
                loss_scale,
            )
        loss_per_token = loss_per_token.reshape(batch_size, -1)
        loss_next = loss_per_token[:, :seq_len]
        mask = labels_next != -100

        loss_unscaled = loss_next
        valid_mask = mask
        if isinstance(loss_scale, torch.Tensor):
            loss_scale_next = loss_scale[:, 1 : seq_len + 1]
            if loss_scale_next.numel() == loss_next.numel():
                loss_scale_next = loss_scale_next.to(
                    device=loss_next.device, dtype=loss_next.dtype
                )
                scale_mask = loss_scale_next > 0
                valid_mask = valid_mask & scale_mask
                if scale_mask.any().item():
                    loss_unscaled = loss_next.clone()
                    loss_unscaled[scale_mask] = (
                        loss_next[scale_mask] / loss_scale_next[scale_mask]
                    )

        if valid_mask.any().item() and loss_unscaled[valid_mask].sum().item() == 0.0:
            return self._split_token_ce_from_logits(
                logits_next,
                labels_next,
                coord_positions_mask,
                num_items_in_batch,
                loss_scale,
            )

        coord_mask = coord_positions_mask & valid_mask
        desc_mask = (~coord_positions_mask) & valid_mask

        coord_ce = (
            loss_unscaled[coord_mask].mean()
            if coord_mask.any().item()
            else loss_next.new_tensor(0.0)
        )
        desc_ce = (
            loss_unscaled[desc_mask].mean()
            if desc_mask.any().item()
            else loss_next.new_tensor(0.0)
        )

        return coord_ce, desc_ce

    def _split_token_ce_from_logits(
        self,
        logits_next: torch.Tensor,
        labels_next: torch.Tensor,
        coord_positions_mask: torch.Tensor,
        num_items_in_batch: int | None,
        loss_scale: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits_next is None or labels_next is None:
            zero = torch.tensor(0.0)
            return zero, zero

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
                loss_scale_next = loss_scale_next.to(
                    device=ce.device, dtype=ce.dtype
                )
                valid_mask = valid_mask & (loss_scale_next > 0)

        coord_mask = coord_positions_mask & valid_mask
        desc_mask = (~coord_positions_mask) & valid_mask

        coord_ce = (
            ce[coord_mask].mean()
            if coord_mask.any().item()
            else ce.new_tensor(0.0)
        )
        desc_ce = (
            ce[desc_mask].mean()
            if desc_mask.any().item()
            else ce.new_tensor(0.0)
        )

        return coord_ce, desc_ce

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
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0.0, 1.0
        )
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(
            0.0, 1.0
        )
        pred_mask = soft_polygon_mask(
            pred,
            mask_size=mask_size,
            sigma_mask=sigma_mask,
            tau_inside=tau_inside,
            beta_dist=beta_dist,
            eps=eps,
        )
        target_mask = soft_polygon_mask(
            target,
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
        return inter / (union + eps)

    @staticmethod
    def _poly_smoothness(vertices: torch.Tensor) -> torch.Tensor:
        vertices = torch.nan_to_num(
            vertices, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
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
