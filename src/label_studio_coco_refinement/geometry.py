"""Editing-view conversions for the inclusive norm1000 ``xyxy`` lattice."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from src.common.errors import DataContractError
from src.data.geometry import coord_bins_to_pixel_xyxy, validate_bbox_bins


NormBBox = tuple[int, int, int, int]
FloatBBox = tuple[float, float, float, float]
LabelStudioXYWH = tuple[float, float, float, float]

NORM1000_MAX = 999
OUTWARD_BIN_TOLERANCE = 1e-9
UNCHANGED_PERCENT_TOLERANCE = 1e-12


def norm1000_edge_to_percent(value: Any) -> float:
    """Map one inclusive ``0..999`` edge bin directly to percentage space."""

    return 100.0 * _validate_edge_bin(value) / NORM1000_MAX


def norm1000_bbox_to_label_studio_xywh(value: Any) -> LabelStudioXYWH:
    """Convert strict norm1000 ``xyxy`` to Label Studio percentage ``xywh``."""

    x1, y1, x2, y2 = validate_bbox_bins(value, field="bbox_2d")
    left = norm1000_edge_to_percent(x1)
    top = norm1000_edge_to_percent(y1)
    right = norm1000_edge_to_percent(x2)
    bottom = norm1000_edge_to_percent(y2)
    return left, top, right - left, bottom - top


def outward_quantize_norm1000_xyxy(value: Any) -> NormBBox:
    """Outward-quantize percentage ``xyxy`` with the approved float tolerance."""

    left, top, right, bottom = _finite_four(value, field="percent_xyxy")
    quantized = (
        math.floor(left * NORM1000_MAX / 100.0 + OUTWARD_BIN_TOLERANCE),
        math.floor(top * NORM1000_MAX / 100.0 + OUTWARD_BIN_TOLERANCE),
        math.ceil(right * NORM1000_MAX / 100.0 - OUTWARD_BIN_TOLERANCE),
        math.ceil(bottom * NORM1000_MAX / 100.0 - OUTWARD_BIN_TOLERANCE),
    )
    clipped = tuple(min(NORM1000_MAX, max(0, edge)) for edge in quantized)
    return validate_bbox_bins(clipped, field="quantized_bbox_2d")


def label_studio_xywh_to_norm1000(
    x: Any,
    y: Any,
    width: Any,
    height: Any,
    *,
    previous_bbox: Any | None = None,
) -> NormBBox:
    """Convert Label Studio percentage ``xywh`` to strict norm1000 ``xyxy``.

    When the serialized percentage geometry still represents ``previous_bbox``,
    those exact committed integers are reused. Otherwise the edited rectangle is
    tolerance-aware outward-quantized.
    """

    values = _finite_four((x, y, width, height), field="label_studio_xywh")
    if previous_bbox is not None:
        previous = validate_bbox_bins(previous_bbox, field="previous_bbox_2d")
        expected = norm1000_bbox_to_label_studio_xywh(previous)
        if all(
            math.isclose(actual, prior, rel_tol=0.0, abs_tol=UNCHANGED_PERCENT_TOLERANCE)
            for actual, prior in zip(values, expected, strict=True)
        ):
            return previous
    left, top, rectangle_width, rectangle_height = values
    if rectangle_width <= 0.0 or rectangle_height <= 0.0:
        raise DataContractError(
            "Label Studio rectangle must have positive width and height",
            code="label_studio.rectangle_order",
            context={"width": rectangle_width, "height": rectangle_height},
        )
    return outward_quantize_norm1000_xyxy(
        (left, top, left + rectangle_width, top + rectangle_height)
    )


def parser_bins_to_canvas_xyxy(
    value: Any,
    *,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    """Execute the current parser's distinct ``round(bin*extent/1000)`` rule."""

    return coord_bins_to_pixel_xyxy(
        value,
        image_width=canvas_width,
        image_height=canvas_height,
        field="parser_canvas_bbox",
    )


def pixel_xyxy_to_norm1000(
    value: Any,
    *,
    image_width: int,
    image_height: int,
) -> NormBBox:
    """Clip original-image pixel edges and outward-quantize to norm1000."""

    width = _positive_extent(image_width, field="image_width")
    height = _positive_extent(image_height, field="image_height")
    x1, y1, x2, y2 = _finite_four(value, field="pixel_xyxy")
    clipped = (
        min(float(width), max(0.0, x1)),
        min(float(height), max(0.0, y1)),
        min(float(width), max(0.0, x2)),
        min(float(height), max(0.0, y2)),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise DataContractError(
            "pixel bbox is degenerate after image clipping",
            code="label_studio.pixel_bbox_order",
            context={"bbox": list(clipped), "image_width": width, "image_height": height},
        )
    return outward_quantize_norm1000_xyxy(
        (
            100.0 * clipped[0] / width,
            100.0 * clipped[1] / height,
            100.0 * clipped[2] / width,
            100.0 * clipped[3] / height,
        )
    )


def _validate_edge_bin(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(
            "norm1000 edge must be an integer",
            code="label_studio.norm_edge_type",
            context={"value": value, "value_type": type(value).__name__},
        )
    if value < 0 or value > NORM1000_MAX:
        raise DataContractError(
            "norm1000 edge is outside the inclusive lattice",
            code="label_studio.norm_edge_range",
            context={"value": value, "min": 0, "max": NORM1000_MAX},
        )
    return value


def _finite_four(value: Any, *, field: str) -> FloatBBox:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise DataContractError(
            "geometry must contain exactly four numeric values",
            code="label_studio.geometry_shape",
            context={"field": field},
        )
    parsed: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise DataContractError(
                "geometry values must be numeric",
                code="label_studio.geometry_type",
                context={"field": f"{field}[{index}]", "value_type": type(item).__name__},
            )
        number = float(item)
        if not math.isfinite(number):
            raise DataContractError(
                "geometry values must be finite",
                code="label_studio.geometry_finite",
                context={"field": f"{field}[{index}]", "value": repr(number)},
            )
        parsed.append(number)
    return parsed[0], parsed[1], parsed[2], parsed[3]


def _positive_extent(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataContractError(
            "image extent must be a positive integer",
            code="label_studio.image_extent",
            context={"field": field, "value": value},
        )
    return value


__all__ = [
    "FloatBBox",
    "LabelStudioXYWH",
    "NORM1000_MAX",
    "NormBBox",
    "OUTWARD_BIN_TOLERANCE",
    "UNCHANGED_PERCENT_TOLERANCE",
    "label_studio_xywh_to_norm1000",
    "norm1000_bbox_to_label_studio_xywh",
    "norm1000_edge_to_percent",
    "outward_quantize_norm1000_xyxy",
    "parser_bins_to_canvas_xyxy",
    "pixel_xyxy_to_norm1000",
]
