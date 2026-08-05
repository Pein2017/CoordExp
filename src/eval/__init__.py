"""Evaluation entrypoints.

The package root keeps exports lazy so an offline detection artifact reducer does
not import forward-eval training dependencies just to parse CLI arguments.
"""

__all__ = [
    "EVAL_FORWARD_SPLIT",
    "EVAL_REDUCTION_DISJOINT_SHARD",
    "EVAL_REDUCTION_REPLICATED",
    "ForwardEvalObservation",
    "ForwardEvalRunner",
    "partition_eval_micro_steps_for_rank",
    "resolve_active_eval_reduction_mode",
    "resolve_eval_reduction_control",
    "METRIC_FAMILY",
    "DetectionConsumerResult",
    "evaluate_scored_detection_artifacts",
]

_FORWARD_NAMES = {
    "EVAL_FORWARD_SPLIT",
    "EVAL_REDUCTION_DISJOINT_SHARD",
    "EVAL_REDUCTION_REPLICATED",
    "ForwardEvalObservation",
    "ForwardEvalRunner",
    "partition_eval_micro_steps_for_rank",
    "resolve_active_eval_reduction_mode",
    "resolve_eval_reduction_control",
}


def __getattr__(name: str) -> object:
    if name in {
        "METRIC_FAMILY",
        "DetectionConsumerResult",
        "evaluate_scored_detection_artifacts",
    }:
        from src.eval import detection_consumer

        return getattr(detection_consumer, name)
    if name in _FORWARD_NAMES:
        from src.eval import forward

        return getattr(forward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
