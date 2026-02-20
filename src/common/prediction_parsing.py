"""Prediction parsing + coordinate heuristics shared across inference/eval."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from src.common.geometry.coord_utils import (
    MAX_BIN,
    coerce_point_list,
    flatten_points,
    ints_to_pixels_norm1000,
    pair_points,
)
from src.utils.coordjson_transpiler import coordjson_to_strict_json_with_meta

GEOM_KEYS = ("bbox_2d", "poly")

_SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")


def extract_special_tokens(text: str) -> List[str]:
    """Extract special tokens like ``<|im_end|>`` / ``<|coord_123|>`` in order."""

    out: List[str] = []
    seen: set[str] = set()
    for match in _SPECIAL_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def extract_json_block(text: str) -> str | None:
    """Return the first balanced top-level JSON object substring."""

    n = len(text)
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, n):
            cur = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    in_string = False
                continue
            if cur == '"':
                in_string = True
                continue
            if cur == "{":
                depth += 1
                continue
            if cur == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def load_prediction_dict(text: str) -> Dict[str, Any] | None:
    """Load model output into strict JSON dict ``{"objects": [...]}``.

    The model-facing output is CoordJSON, so this helper first runs the
    CoordJSON -> strict-JSON transpiler (salvage mode) and then loads JSON via
    ``json.loads``.

    Eval/infer is intentionally order-salvage oriented: it tries both
    ``geometry_first`` and ``desc_first`` parsing policies and keeps the payload
    with the largest retained ``objects`` list. This intentionally does not
    enforce ``custom.object_field_order``.
    """

    best_payload: Dict[str, Any] | None = None
    best_count = -1

    for order in ("geometry_first", "desc_first"):
        try:
            strict_text, meta = coordjson_to_strict_json_with_meta(
                text,
                mode="salvage",
                object_field_order=order,
            )
            if bool(meta.parse_failed):
                continue
            parsed = json.loads(strict_text)
        except (ValueError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        objects = parsed.get("objects")
        if not isinstance(objects, list):
            continue
        count = int(len(objects))
        if count > best_count:
            best_payload = parsed
            best_count = count

    return best_payload


def parse_prediction(text: str) -> List[Dict[str, Any]]:
    """Parse model output JSON into a list of objects with integer coords.

    Primary path: CoordJSON salvage -> strict JSON via ``load_prediction_dict``.

    Fallback path (legacy/unit tests): if salvage yields no payload, attempt to
    parse a raw JSON object block directly.
    """

    obj = load_prediction_dict(text)
    if obj is None:
        block = extract_json_block(text) or text
        try:
            parsed_raw = json.loads(block)
        except (ValueError, TypeError):
            return []
        if not isinstance(parsed_raw, dict):
            return []
        obj = parsed_raw

    objects_raw = obj.get("objects")
    if isinstance(objects_raw, list):
        objects = objects_raw
    else:
        # Legacy single-object schemas (used in some older checkpoints / tests).
        if isinstance(obj.get("obj"), dict):
            objects = [obj["obj"]]
        elif isinstance(obj.get("object"), dict):
            objects = [obj["object"]]
        elif any(g in obj for g in GEOM_KEYS) or "line" in obj or "line_points" in obj:
            objects = [obj]
        else:
            return []

    parsed: List[Dict[str, Any]] = []
    for entry in objects:
        if not isinstance(entry, dict):
            continue

        if "line" in entry or "line_points" in entry:
            parsed.append(
                {
                    "desc": str(entry.get("desc", "")),
                    "line": entry.get("line"),
                    "line_points": entry.get("line_points"),
                }
            )
            continue

        geom_keys = [g for g in GEOM_KEYS if g in entry]
        if len(geom_keys) != 1:
            continue
        gtype = geom_keys[0]

        pts_raw = flatten_points(entry.get(gtype))
        if pts_raw is None or len(pts_raw) % 2 != 0:
            continue

        points, had_tokens = coerce_point_list(pts_raw)
        if points is None:
            continue

        if had_tokens and any(v < 0 or v > MAX_BIN for v in points):
            continue

        ints = [int(round(v)) for v in points]
        parsed.append(
            {
                "desc": str(entry.get("desc", "")),
                "type": gtype,
                "points": ints,
                "_had_tokens": had_tokens,
            }
        )

    return parsed


def coords_are_pixel(
    points: Sequence[float], width: float, height: float, *, had_tokens: bool
) -> bool:
    """Heuristic: treat coords as pixel-space when tokens are absent and bins exceed norms."""

    if had_tokens:
        return False
    if not points:
        return False
    if width <= 0 or height <= 0:
        return False

    max_coord = max(points)
    max_wh = max(width, height)

    if max_coord > MAX_BIN:
        return True
    if max_coord > max_wh:
        return True
    if max_wh <= MAX_BIN:
        return True

    return False


def clamp_points(points: Sequence[float], width: float, height: float) -> List[float]:
    """Clamp coordinates to image bounds (float-preserving)."""

    out: List[float] = []
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    for i, value in enumerate(points):
        bound = w - 1.0 if i % 2 == 0 else h - 1.0
        out.append(min(max(float(value), 0.0), bound))
    return out


def ints_to_pixels(ints: Sequence[int], width: float, height: float) -> List[float]:
    """Convert normalized 0-999 coords to pixel coordinates."""

    return ints_to_pixels_norm1000(ints, width, height)


__all__ = [
    "GEOM_KEYS",
    "MAX_BIN",
    "extract_special_tokens",
    "load_prediction_dict",
    "extract_json_block",
    "parse_prediction",
    "coords_are_pixel",
    "clamp_points",
    "ints_to_pixels",
    "pair_points",
]

