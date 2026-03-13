"""Metric-key helpers for Stage-2 eval namespaces.

Hugging Face best-checkpoint selection assumes evaluation metrics live under
`eval_<metric_for_best_model>`. Our Stage-2 evaluator instead emits grouped keys
such as `eval/detection/mAP` and `eval/runtime/coco_eval_ok`. These helpers keep
the emitted metrics stable while letting checkpoint-selection code resolve the
slash-namespaced variants.
"""

from __future__ import annotations

from typing import Any, Mapping

_DETECTION_KEYS = frozenset(
    {
        "precision",
        "recall",
        "f1",
        "pred_objects",
        "gt_objects_total",
        "matched",
        "fp_total",
        "fn_total",
        "matched_maskiou_mean",
        "sample_any_match_rate",
        "mAP",
    }
)
_PARSING_KEYS = frozenset(
    {
        "gating_rejections",
        "parse_dropped_invalid",
        "parse_dropped_ambiguous",
        "parse_truncated_rate",
        "sample_valid_pred_rate",
    }
)
_DESCRIPTION_KEYS = frozenset(
    {
        "desc_pairs_total",
        "desc_exact_acc_on_matched",
        "desc_sem_enabled",
        "desc_sem_acc_on_matched",
        "desc_sem_sim_mean",
        "desc_sem_sim_count",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "prompt_variant_is_coco_80",
        "effective_score_mode_is_constant",
        "effective_score_mode_is_confidence_postop",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "trace_fallback_count",
        "vllm_decode_error_count",
        "coco_eval_ok",
    }
)
_DIRECT_EVAL_GROUPS = frozenset(
    {"detection", "parsing", "description", "config", "runtime"}
)


def stage2_eval_metric_key(metric_key_prefix: str, suffix: str) -> str:
    """Return the Stage-2 eval metric key for a rollout/time suffix."""

    prefix = str(metric_key_prefix)
    leaf_suffix = str(suffix)
    if prefix == "eval":
        if leaf_suffix.startswith("time/"):
            leaf = leaf_suffix[len("time/") :]
            return f"eval/runtime/{leaf}"
        if leaf_suffix.startswith("rollout/"):
            leaf = leaf_suffix[len("rollout/") :]
            if leaf in _DETECTION_KEYS:
                return f"eval/detection/{leaf}"
            if leaf in _PARSING_KEYS:
                return f"eval/parsing/{leaf}"
            if leaf in _DESCRIPTION_KEYS:
                return f"eval/description/{leaf}"
            if leaf in _CONFIG_KEYS:
                return f"eval/config/{leaf}"
            if leaf in _RUNTIME_KEYS or leaf.startswith("coco_counter_"):
                return f"eval/runtime/{leaf}"
            return f"eval/runtime/{leaf}"
    return f"{prefix}_{leaf_suffix}"


def metric_lookup_candidates(metric_name: str) -> tuple[str, ...]:
    """Return candidate metric keys that may represent the same metric.

    This bridges:
    - stock transformers keys like `eval_detection/mAP`
    - Stage-2 grouped eval keys like `eval/detection/mAP`
    - Stage-2 config values like `rollout/f1` and `detection/mAP`
    """

    raw = str(metric_name or "").strip()
    if not raw:
        return ()

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(key: str | None) -> None:
        if not key:
            return
        if key in seen:
            return
        seen.add(key)
        ordered.append(key)

    _add(raw)

    if raw.startswith("eval_"):
        remainder = raw[len("eval_") :]
        if "/" in remainder:
            group, leaf = remainder.split("/", 1)
            _add(f"eval/{group}/{leaf}")
        if remainder.startswith("rollout/"):
            _add(stage2_eval_metric_key("eval", remainder))
        return tuple(ordered)

    if raw.startswith("eval/"):
        remainder = raw[len("eval/") :]
        if "/" in remainder:
            group, leaf = remainder.split("/", 1)
            _add(f"eval_{group}/{leaf}")
        if remainder.startswith("rollout/"):
            _add(stage2_eval_metric_key("eval", remainder))
        return tuple(ordered)

    _add(f"eval_{raw}")

    if raw.startswith("rollout/"):
        _add(stage2_eval_metric_key("eval", raw))
    elif "/" in raw:
        group, _leaf = raw.split("/", 1)
        if group in _DIRECT_EVAL_GROUPS:
            _add(f"eval/{raw}")

    return tuple(ordered)


def resolve_metric_value(
    metrics: Mapping[str, Any], metric_name: str
) -> tuple[str, Any] | None:
    """Resolve a metric name against a metrics mapping using known aliases."""

    for key in metric_lookup_candidates(metric_name):
        if key in metrics:
            return key, metrics[key]
    return None


def metric_name_matches_key(metric_name: str, metric_key: str) -> bool:
    """Return True when a metric name can resolve to the provided key."""

    return str(metric_key) in metric_lookup_candidates(metric_name)
