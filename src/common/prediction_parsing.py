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

    Accepts 3 common shapes:

    1) Strict JSON: ``{"objects": [...]}``.
    2) Legacy JSON: an index-keyed dict like ``{"0": {...}, "1": {...}}``.
       This is normalized to strict JSON by sorting keys and mapping values to
       ``objects``.
    3) CoordJSON: model-facing format that is transpiled into strict JSON in
       salvage mode.

    Eval/infer is intentionally order-salvage oriented: it tries both
    ``geometry_first`` and ``desc_first`` parsing policies and keeps the payload
    with the largest retained ``objects`` list. This intentionally does not
    enforce ``custom.object_field_order``.

    This helper is best-effort: model-output parse failures are represented as
    ``None`` so inference can continue-but-observable at the sample level.
    """

    json_block = extract_json_block(text)
    if json_block is not None:
        try:
            parsed = json.loads(json_block)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            objects = parsed.get("objects")
            if isinstance(objects, list):
                return parsed

            # Common legacy shape: {"0": {..obj..}, "1": {..obj..}}
            if parsed and all(isinstance(k, str) and k.isdigit() for k in parsed.keys()):
                keyed: list[tuple[int, Any]] = []
                for k, v in parsed.items():
                    if isinstance(k, str) and k.isdigit():
                        keyed.append((int(k), v))
                keyed.sort(key=lambda kv: kv[0])
                legacy_objects = [v for _idx, v in keyed if isinstance(v, dict)]
                if legacy_objects:
                    return {"objects": legacy_objects}

            # Legacy object-map shape: {"obj": {...}, "obj2": {...}}
            if parsed and all(isinstance(v, dict) for v in parsed.values()):
                legacy_objects = [v for v in parsed.values() if isinstance(v, dict)]
                if legacy_objects:
                    return {"objects": legacy_objects}

            # Single-object dict at top-level: {"bbox_2d": ..., "desc": ...}
            if any(k in parsed for k in GEOM_KEYS) or "line" in parsed or "line_points" in parsed:
                return {"objects": [parsed]}

    best_payload: Dict[str, Any] | None = None
    best_count = -1

    for order in ("geometry_first", "desc_first"):
        strict_text, meta = coordjson_to_strict_json_with_meta(
            text,
            mode="salvage",
            object_field_order=order,
        )
        if bool(meta.parse_failed):
            continue
        try:
            parsed = json.loads(strict_text)
        except json.JSONDecodeError:
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
    """Parse model output JSON into a list of objects with integer coords."""

    obj = load_prediction_dict(text)
    if obj is None:
        return []

    objects = obj.get("objects")
    if not isinstance(objects, list):
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

