"""Geometry and coordinate-token helpers owned by ``public_data``."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

COORD_MIN = 0
COORD_MAX = 999
MAX_BIN = COORD_MAX
COORD_TOKEN_RE = re.compile(r"^<\|coord_(0|[1-9][0-9]{0,2})\|>$")

CXCYWH_SLOT_ORDER = ("cx", "cy", "w", "h")
CXCYWH_CONVERSION_VERSION = "norm1000-cxcywh-v1"
CXCY_LOGW_LOGH_SLOT_ORDER = ("cx", "cy", "log_w", "log_h")
CXCY_LOGW_LOGH_CONVERSION_VERSION = "norm1000-cxcy-logw-logh-v1"


def is_coord_token(value: object) -> bool:
    return isinstance(value, str) and COORD_TOKEN_RE.fullmatch(value) is not None


def token_to_int(value: str) -> int:
    match = COORD_TOKEN_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid coordinate token: {value!r}")
    return int(match.group(1))


def int_to_token(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or not COORD_MIN <= value <= COORD_MAX:
        raise ValueError(f"coordinate bin must be an integer in [0, 999], got {value!r}")
    return f"<|coord_{value}|>"


def tokens_to_ints(values: Sequence[object]) -> list[int]:
    return [token_to_int(str(value)) for value in values]


def normalize_bbox_format(value: str | None, *, path: str | None = None) -> str:
    normalized = str(value or "xyxy").strip().lower().replace("-", "_")
    aliases = {"xyxy": "xyxy", "cxcywh": "cxcywh", "cxcy_logw_logh": "cxcy_logw_logh"}
    if normalized not in aliases:
        label = path or "bbox_format"
        raise ValueError(f"unsupported {label}: {value!r}")
    return aliases[normalized]


def clamp_points(values: Sequence[float], width: int, height: int) -> list[float]:
    return [
        min(max(float(value), 0.0), max(0.0, float(width if index % 2 == 0 else height)))
        for index, value in enumerate(values)
    ]


def scale_points(values: Sequence[float], scale_x: float, scale_y: float) -> list[float]:
    return [float(value) * (scale_x if index % 2 == 0 else scale_y) for index, value in enumerate(values)]


def round_points(values: Sequence[float]) -> list[int]:
    return [int(round(float(value))) for value in values]


def ints_to_pixels_norm1000(values: Sequence[int], width: int, height: int) -> list[int]:
    return [
        int(round(int(value) * (width if index % 2 == 0 else height) / 1000.0))
        for index, value in enumerate(values)
    ]


def xyxy_norm1000_to_cxcywh_bins(values: Sequence[int]) -> list[int]:
    x1, y1, x2, y2 = _validated_xyxy(values)
    return [_bin((x1 + x2) / 2), _bin((y1 + y2) / 2), _bin(x2 - x1), _bin(y2 - y1)]


def cxcywh_norm1000_to_xyxy_norm1000(values: Sequence[int]) -> list[int]:
    cx, cy, width, height = _four(values)
    return [_bin(cx - width / 2), _bin(cy - height / 2), _bin(cx + width / 2), _bin(cy + height / 2)]


def xyxy_norm1000_to_cxcy_logw_logh_bins(values: Sequence[int]) -> list[int]:
    x1, y1, x2, y2 = _validated_xyxy(values)
    return [
        _bin((x1 + x2) / 2),
        _bin((y1 + y2) / 2),
        _encode_log_extent(x2 - x1),
        _encode_log_extent(y2 - y1),
    ]


def cxcy_logw_logh_norm1000_to_xyxy_norm1000(values: Sequence[int]) -> list[int]:
    cx, cy, log_width, log_height = _four(values)
    width = _decode_log_extent(log_width)
    height = _decode_log_extent(log_height)
    return [_bin(cx - width / 2), _bin(cy - height / 2), _bin(cx + width / 2), _bin(cy + height / 2)]


def _encode_log_extent(extent: int) -> int:
    return _bin(math.log1p(max(1, extent)) / math.log1p(999) * 999)


def _decode_log_extent(value: int) -> float:
    return math.expm1(_bin(value) / 999 * math.log1p(999))


def _bin(value: float | int) -> int:
    return max(COORD_MIN, min(COORD_MAX, int(round(float(value)))))


def _four(values: Sequence[int]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError(f"expected four coordinate values, got {len(values)}")
    return tuple(_bin(value) for value in values)  # type: ignore[return-value]


def _validated_xyxy(values: Sequence[int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _four(values)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"expected non-degenerate xyxy coordinates, got {list(values)!r}")
    return x1, y1, x2, y2
