"""Rollout-matching telemetry contracts.

This module contains data structures used to buffer/aggregate rollout-matching
training diagnostics across micro-batches.

It is intentionally independent from trainer implementation modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


def slim_rollout_meta_for_logging(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop large fields (e.g. token id lists) from rollout meta before buffering for logs.

    Rollout-matching uses gradient accumulation; we want ONE log point per optimizer step.
    We therefore buffer meta across micro-batches, but keep it lightweight.
    """

    def _as_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _as_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    out: Dict[str, Any] = {
        "prompt_len": _as_int(meta.get("prompt_len", 0)),
        "rollout_len": _as_int(meta.get("rollout_len", 0)),
        "prefix_len": _as_int(meta.get("prefix_len", 0)),
        "train_len": _as_int(meta.get("train_len", 0)),
        "encoded_len": _as_int(meta.get("encoded_len", 0)),
        "decode_mode": str(meta.get("decode_mode", "")),
        "parse_dropped_invalid": _as_int(meta.get("parse_dropped_invalid", 0)),
        "parse_dropped_ambiguous": _as_int(meta.get("parse_dropped_ambiguous", 0)),
        "parse_truncated": bool(meta.get("parse_truncated", False)),
        "valid_pred_objects": _as_int(meta.get("valid_pred_objects", 0)),
        "matched_for_supervision": _as_int(meta.get("matched_for_supervision", 0)),
        "matched_maskiou_sum": _as_float(meta.get("matched_maskiou_sum", 0.0)),
        "matched_maskiou_count": _as_int(meta.get("matched_maskiou_count", 0)),
        "gt_objects": _as_int(meta.get("gt_objects", 0)),
        "fn_count": _as_int(meta.get("fn_count", 0)),
        "gating_rejections": _as_int(meta.get("gating_rejections", 0)),
        "excluded_from_supervision": _as_int(meta.get("excluded_from_supervision", 0)),
        # For token masking diagnostics.
        "prefix_coord_target_bins": list(meta.get("prefix_coord_target_bins") or []),
        "tail_ignore_pos": list(meta.get("tail_ignore_pos") or []),
        # Desc monitor fields (optional).
        "desc_monitor_ran": bool(meta.get("desc_monitor_ran", False)),
        "desc_pairs_total": _as_int(meta.get("desc_pairs_total", 0)),
        "desc_exact_ok": _as_int(meta.get("desc_exact_ok", 0)),
        "desc_sem_ok": _as_int(meta.get("desc_sem_ok", 0)),
        "desc_sem_sim_sum": _as_float(meta.get("desc_sem_sim_sum", 0.0)),
        "desc_sem_sim_count": _as_int(meta.get("desc_sem_sim_count", 0)),
        "desc_sem_enabled": _as_int(meta.get("desc_sem_enabled", 0)),
    }

    return out


@dataclass
class PendingTrainRolloutLog:
    """Accumulate rollout-matching logs across micro-batches for ONE optimizer step."""

    meta: List[Dict[str, Any]] = field(default_factory=list)

    ce_loss_sum: float = 0.0
    coord_loss_sum: float = 0.0
    coord_prefix_sum: float = 0.0
    coord_tail_sum: float = 0.0
    n_micro: int = 0

    time_forward_s: float = 0.0
    time_mask_build_s: float = 0.0

    # Rollout pipeline timings.
    time_rollout_generate_s: float = 0.0
    time_rollout_parse_match_s: float = 0.0
    time_rollout_teacher_encode_s: float = 0.0

    # Packing/collation timing.
    time_post_rollout_pack_s: float = 0.0

    # Packing stats (optional; only populated when packing is enabled).
    packing_fill_sum: float = 0.0
    packing_selected_total_len_sum: float = 0.0
    packing_segments_sum: float = 0.0
    packing_count: int = 0
    packing_buffer_last: float = 0.0

    def add_micro(
        self,
        *,
        meta: List[Mapping[str, Any]],
        ce_loss: float,
        coord_loss: float,
        coord_prefix: float,
        coord_tail: float,
        time_forward_s: float,
        time_mask_build_s: float,
        batch_metrics: Optional[Mapping[str, Any]],
    ) -> None:
        self.n_micro += 1
        self.ce_loss_sum += float(ce_loss)
        self.coord_loss_sum += float(coord_loss)
        self.coord_prefix_sum += float(coord_prefix)
        self.coord_tail_sum += float(coord_tail)
        self.time_forward_s += float(time_forward_s)
        self.time_mask_build_s += float(time_mask_build_s)

        for m in meta:
            if isinstance(m, Mapping):
                self.meta.append(slim_rollout_meta_for_logging(m))

        if not isinstance(batch_metrics, Mapping):
            return

        # Timings.
        self.time_rollout_generate_s += float(
            batch_metrics.get("time/rollout_generate_s", 0.0) or 0.0
        )
        self.time_rollout_parse_match_s += float(
            batch_metrics.get("time/rollout_parse_match_s", 0.0) or 0.0
        )
        self.time_rollout_teacher_encode_s += float(
            batch_metrics.get("time/rollout_teacher_encode_s", 0.0) or 0.0
        )
        self.time_post_rollout_pack_s += float(
            batch_metrics.get("time/post_rollout_pack_s", 0.0) or 0.0
        )

        # Packing stats.
        if "packing/post_rollout_fill" in batch_metrics:
            self.packing_fill_sum += float(
                batch_metrics.get("packing/post_rollout_fill", 0.0) or 0.0
            )
            self.packing_selected_total_len_sum += float(
                batch_metrics.get("packing/post_rollout_selected_total_len", 0.0) or 0.0
            )
            self.packing_segments_sum += float(
                batch_metrics.get("packing/post_rollout_segments", 0.0) or 0.0
            )
            self.packing_buffer_last = float(
                batch_metrics.get(
                    "packing/post_rollout_buffer", self.packing_buffer_last
                )
                or self.packing_buffer_last
            )
            self.packing_count += 1


__all__ = ["slim_rollout_meta_for_logging", "PendingTrainRolloutLog"]
