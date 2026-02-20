"""
Evaluation helpers for CoordExp.

Exports shared parsing utilities and detection evaluator entrypoints.
"""

from .detection import EvalOptions, evaluate_and_save, evaluate_detection
from .bbox_confidence import compute_bbox_confidence_from_logprobs
from .confidence_postop import (
    ConfidencePostOpPaths,
    run_confidence_postop,
    run_confidence_postop_from_config,
)
from src.common.geometry.coord_utils import COORD_TOKEN_RE
from .parsing import (
    GEOM_KEYS,
    MAX_BIN,
    clamp_points,
    coords_are_pixel,
    extract_json_block,
    ints_to_pixels,
    pair_points,
    parse_prediction,
)

__all__ = [
    "EvalOptions",
    "evaluate_detection",
    "evaluate_and_save",
    "compute_bbox_confidence_from_logprobs",
    "ConfidencePostOpPaths",
    "run_confidence_postop",
    "run_confidence_postop_from_config",
    "COORD_TOKEN_RE",
    "GEOM_KEYS",
    "MAX_BIN",
    "clamp_points",
    "coords_are_pixel",
    "extract_json_block",
    "ints_to_pixels",
    "parse_prediction",
    "pair_points",
]
