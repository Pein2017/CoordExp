from typing import Any, Mapping

import torch


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

        loss = loss.float() + result.total_loss.float()
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
        if has_total:
            reporter.update(
                "stage1/total_loss_per_sample_est",
                float(total_est) + float(bbox_loss_per_sample),
            )
        setattr(self, "_coordexp_last_bbox_geo_loss_per_sample", float(bbox_loss_per_sample))

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

        loss = loss.float() + result.total_loss.float()
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


__all__ = [
    "BBoxGeoLossMixin",
    "BBoxSizeAuxLossMixin",
]
