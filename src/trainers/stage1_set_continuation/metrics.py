from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.metrics.reporter import SwiftMetricReporter


STAGE1_SET_CONTINUATION_METRIC_SCHEMA_VERSION = "stage1_set_continuation_metrics_v2"

EMITTED_STAGE1_SET_CONTINUATION_METRICS = {
    "loss/candidate_balanced",
    "loss/coord_gate",
    "loss/schema_open",
    "loss/text_gate",
    "loss/json_structural",
    "loss/anti_close_start",
    "loss/weak_schema_close",
    "loss/rmp",
    "loss/rmp_branch_ce",
    "loss/rmp_unique_ce",
    "loss/rmp_coord_branch_ce",
    "loss/rmp_desc_text_branch_ce",
    "loss/rmp_boundary_ce",
    "loss/rmp_close_ce",
    "loss/rmp_eos_ce",
    "gate/coord_slot_coord_mass_mean",
    "gate/text_slot_coord_mass_mean",
    "gate/coord_tokens_count",
    "gate/text_tokens_count",
    "mp/num_prefix_objects",
    "mp/num_remaining_objects",
    "mp/num_candidates_scored",
    "mp/candidate_tokens_scored_mean",
    "mp/schema_open_tokens_scored_mean",
    "mp/json_structural_tokens_scored_mean",
    "mp/annotation_completeness_weight_mean",
    "mp/final_close_weight_mean",
    "mp/tail_positive_samples",
    "mp/final_gt_object_scored_samples",
    "mp/objective_fidelity_exact_samples",
    "mp/fallback_applied_samples",
    "mp/selected_mode_empty_prefix",
    "mp/selected_mode_random_subset",
    "mp/selected_mode_leave_one_out",
    "mp/selected_mode_full_prefix",
    "mp/objective_contributing_samples",
    "rmp/branch_nodes",
    "rmp/branch_nodes_desc_text",
    "rmp/branch_nodes_coord",
    "rmp/branch_nodes_structural",
    "rmp/branch_nodes_other",
    "rmp/valid_children_mean",
    "rmp/target_entropy_mean",
    "rmp/valid_child_mass_mean",
    "rmp/teacher_branch_top1_acc",
    "rmp/valid_child_top1_acc",
    "rmp/gt_count_ge7_samples",
    "stop/p_close_start_when_remaining_exists",
    "stop/p_continue_start_when_remaining_exists",
    "stop/p_close_start_when_remaining_empty",
}

LOGZ_ESTIMATOR_CODES = {
    "exact": 0.0,
    "sampled_raw": 1.0,
    "uniform_importance": 2.0,
}

CANDIDATE_SCORING_MODE_CODES = {
    "exact": 0.0,
    "uniform_subsample": 1.0,
}

PREFIX_ATTACH_MODE_CODES = {
    "repeated_forward": 0.0,
}

BRANCH_ISOLATION_CODES = {
    "independent_forward": 0.0,
}

PREFIX_GRADIENT_CODES = {
    "non_detached_recomputed_per_branch": 0.0,
}

BRANCH_RUNTIME_MODE_CODES = {
    "retained_graph": 0.0,
    "checkpointed_exact": 1.0,
    "smart_batched_exact": 2.0,
}

DDP_CANDIDATE_PADDING_POLICY_CODES = {
    "max_count": 0.0,
    "none": 1.0,
}

BRANCH_BATCH_SCHEDULER_CODES = {
    "disabled": 0.0,
    "constant_volume": 1.0,
    "deterministic_fallback": 2.0,
}


def metric_code(value: str, mapping: Mapping[str, float], *, metric_name: str) -> float:
    key = str(value or "").strip()
    if key not in mapping:
        raise ValueError(f"{metric_name} has no numeric code for value: {key!r}")
    return float(mapping[key])


def numeric_metric_payload(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep trainer custom metrics scalar-only.

    The provenance layer records string-valued fields such as candidate scoring
    mode; ms-swift's `custom_metrics` surface stores floats.
    """

    payload: dict[str, float] = {}
    for key, value in metrics.items():
        metric_key = str(key)
        if metric_key not in EMITTED_STAGE1_SET_CONTINUATION_METRICS:
            continue
        if isinstance(value, bool):
            payload[metric_key] = float(value)
            continue
        if isinstance(value, int | float):
            numeric = float(value)
            if math.isfinite(numeric):
                payload[metric_key] = numeric
    return payload


def mean_numeric_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        for key, value in numeric_metric_payload(row).items():
            buckets.setdefault(key, []).append(float(value))
    return {
        key: float(sum(values) / len(values))
        for key, values in sorted(buckets.items())
        if values
    }


def emit_stage1_set_continuation_metrics(
    trainer: Any,
    metrics: Mapping[str, Any],
) -> None:
    SwiftMetricReporter(trainer).update_many(numeric_metric_payload(metrics))


__all__ = [
    "BRANCH_ISOLATION_CODES",
    "BRANCH_BATCH_SCHEDULER_CODES",
    "BRANCH_RUNTIME_MODE_CODES",
    "CANDIDATE_SCORING_MODE_CODES",
    "DDP_CANDIDATE_PADDING_POLICY_CODES",
    "EMITTED_STAGE1_SET_CONTINUATION_METRICS",
    "LOGZ_ESTIMATOR_CODES",
    "PREFIX_ATTACH_MODE_CODES",
    "PREFIX_GRADIENT_CODES",
    "STAGE1_SET_CONTINUATION_METRIC_SCHEMA_VERSION",
    "emit_stage1_set_continuation_metrics",
    "metric_code",
    "mean_numeric_metrics",
    "numeric_metric_payload",
]
