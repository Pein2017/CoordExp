"""Evaluation entrypoints.

The package root keeps exports lazy so an offline detection artifact reducer does
not import forward-eval training dependencies just to parse CLI arguments.
"""

__all__ = [
    "EVAL_FORWARD_SPLIT",
    "ForwardEvalObservation",
    "ForwardEvalRunner",
    "METRIC_FAMILY",
    "DetectionConsumerResult",
    "evaluate_scored_detection_artifacts",
]


def __getattr__(name: str) -> object:
    if name in {
        "METRIC_FAMILY",
        "DetectionConsumerResult",
        "evaluate_scored_detection_artifacts",
    }:
        from src.eval import detection_consumer

        return getattr(detection_consumer, name)
    if name in {"EVAL_FORWARD_SPLIT", "ForwardEvalObservation", "ForwardEvalRunner"}:
        from src.eval import forward

        return getattr(forward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
