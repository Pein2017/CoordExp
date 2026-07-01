"""Strict V1 coordinate-bin geometry helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from src.common.errors import DataContractError


COORD_BIN_MIN = 0
COORD_BIN_MAX = 999
COORD_TOKEN_PATTERN = re.compile(r"^<\|coord_(0|[1-9][0-9]{0,2})\|>$")


def parse_coord_token(value: Any, *, field: str) -> int:
    if not isinstance(value, str):
        raise DataContractError(
            "coordinate token must be a string",
            code="data.coord_token_type",
            context={"field": field, "value_type": type(value).__name__},
        )
    match = COORD_TOKEN_PATTERN.fullmatch(value)
    if match is None:
        raise DataContractError(
            "coordinate token is not canonical",
            code="data.coord_token_format",
            context={"field": field, "value": value},
        )
    parsed = int(match.group(1))
    if parsed > COORD_BIN_MAX:
        raise DataContractError(
            "coordinate token is out of range",
            code="data.coord_token_range",
            context={"field": field, "value": value, "max": COORD_BIN_MAX},
        )
    return parsed


def parse_source_bbox_tokens(value: Any, *, field: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DataContractError(
            "source bbox_2d must be a four-token sequence",
            code="data.bbox_shape",
            context={"field": field, "value_type": type(value).__name__},
        )
    if len(value) != 4:
        raise DataContractError(
            "source bbox_2d must have four values",
            code="data.bbox_shape",
            context={"field": field, "length": len(value)},
        )
    return validate_bbox_bins(
        [parse_coord_token(item, field=f"{field}[{index}]") for index, item in enumerate(value)],
        field=field,
    )


def validate_bbox_bins(value: Any, *, field: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DataContractError(
            "bbox must be a four-integer sequence",
            code="data.bbox_shape",
            context={"field": field, "value_type": type(value).__name__},
        )
    if len(value) != 4:
        raise DataContractError(
            "bbox must have four values",
            code="data.bbox_shape",
            context={"field": field, "length": len(value)},
        )

    parsed: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise DataContractError(
                "bbox values must be integer coordinate bins",
                code="data.bbox_value_type",
                context={
                    "field": f"{field}[{index}]",
                    "value": item,
                    "value_type": type(item).__name__,
                },
            )
        if item < COORD_BIN_MIN or item > COORD_BIN_MAX:
            raise DataContractError(
                "bbox value is out of coordinate-bin range",
                code="data.bbox_value_range",
                context={
                    "field": f"{field}[{index}]",
                    "value": item,
                    "min": COORD_BIN_MIN,
                    "max": COORD_BIN_MAX,
                },
            )
        parsed.append(item)

    x1, y1, x2, y2 = parsed
    if x1 >= x2 or y1 >= y2:
        raise DataContractError(
            "bbox must be non-degenerate x1,y1,x2,y2 coordinate bins",
            code="data.bbox_order",
            context={"field": field, "bbox": parsed},
        )
    return x1, y1, x2, y2


__all__ = [
    "COORD_BIN_MAX",
    "COORD_BIN_MIN",
    "COORD_TOKEN_PATTERN",
    "parse_coord_token",
    "parse_source_bbox_tokens",
    "validate_bbox_bins",
]
