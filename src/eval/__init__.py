"""
Evaluation helpers for CoordExp.

Exports shared parsing utilities and detection evaluator entrypoints.
"""

from .detection import EvalOptions, evaluate_and_save, evaluate_detection
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
