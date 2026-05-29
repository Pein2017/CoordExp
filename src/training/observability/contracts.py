"""Shared observability contracts for clean writes and legacy reads."""

from __future__ import annotations

from src.metrics.events import MetricReducer

REMOVED_TRAINING_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "train/optimization/loss_duplicate_burst_unlikelihood",
        "loss/rollout_correction_text/duplicate_burst_unlikelihood",
        "loss_duplicate_burst_unlikelihood_contrib",
        "coord_softce_w1/adjacent_repulsion",
        "coord_diag/adjacent_repulsion",
        "coord_diag/adjacent_repulsion_pair_count",
        "coord_diag/adjacent_repulsion_applied_count",
        "coord_diag/adjacent_repulsion_copy_score_mean",
        "loss/adjacent_repulsion",
        "loss/rollout_correction_coord/adjacent_repulsion",
        "adjacent_repulsion_contrib",
    }
)

_WEIGHTED_LEGACY_GAUGE_KEYS: frozenset[str] = frozenset(
    {
        "stage2_rollout_correction/correction/dup/raw_duplicate_iou_mean",
        "dup/raw/max_desc_count",
        "dup/raw/saturation_rate",
        "dup/raw/duplicate_like_max_cluster_size",
        "dup/raw/desc_entropy",
    }
)
_SUM_LEGACY_SUFFIXES: tuple[str, ...] = (
    "_count",
    "_total",
    "_sum",
    "_num",
    "_den",
)


def legacy_reducer_for_key(key: str) -> MetricReducer:
    """Return the reducer that preserves legacy flat-key aggregation semantics."""

    if key.startswith("stage2_rollout_correction/correction/dup/N_"):
        return "sum"
    if key in _WEIGHTED_LEGACY_GAUGE_KEYS:
        return "weighted_mean"
    if key.startswith("dup/raw/"):
        if key.endswith(_SUM_LEGACY_SUFFIXES):
            return "sum"
        return "weighted_mean"

    return "last"


__all__ = ["REMOVED_TRAINING_METRIC_KEYS", "legacy_reducer_for_key"]
