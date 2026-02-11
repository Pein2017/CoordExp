"""Prediction parsing + coordinate heuristics shared across inference/eval.

This module owns best-effort JSON extraction/parsing for model generations and a
small set of coordinate-mode heuristics (norm1000 vs pixel). Keeping this in
`src.common` avoids coupling inference-time parsing to evaluator internals.

The core contract is:
- `load_prediction_dict`: parse the model's raw JSON dict from a generation string.
- `parse_prediction`: convert that dict into a list of objects with integer points.

Downstream components (inference engine, evaluator, visualization) should reuse
these helpers instead of duplicating parsing/repair logic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from src.common.geometry.coord_utils import (
    MAX_BIN,
    coerce_point_list,
    flatten_points,
    ints_to_pixels_norm1000,
    pair_points,
)

GEOM_KEYS = ("bbox_2d", "poly")

_SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")


def extract_special_tokens(text: str) -> List[str]:
    """Extract special tokens like ``<|im_end|>`` / ``<|coord_123|>`` in order.

    Deduplicates while preserving first-seen order.
    """

    out: List[str] = []
    seen: set[str] = set()
    for m in _SPECIAL_TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def load_prediction_dict(text: str) -> Dict[str, Any] | None:
    """Load the model's prediction JSON dict from a generation string.

    This is a best-effort loader that tries, in order:
    1) the largest balanced JSON block,
    2) salvaging complete top-level entries from a truncated block,
    3) (only if an explicit terminator exists) closing missing braces.

    Returns:
      A dict mapping object ids -> object payloads, or None if nothing usable
      can be recovered.
    """

    block = extract_json_block(text)
    obj: Any | None = None
    if block:
        obj = _load_json_loose(block)

    if obj is None:
        obj = _salvage_top_level_entries(text)

    if obj is None:
        # Final fallback: heuristically close braces only when the model emitted
        # an explicit terminator (e.g. "<im_end>"). If truncated without a
        # terminator, prefer returning None.
        if "<im_end>" in text and "{" in text:
            start = text.find("{")
            candidate = text[start:]
            missing = candidate.count("{") - candidate.count("}")
            if missing > 0:
                obj = _load_json_loose(candidate + ("}" * missing))

    if isinstance(obj, dict):
        return obj
    return None


def _salvage_top_level_entries(text: str) -> Dict[str, Any] | None:
    """Best-effort salvage of top-level ``{key: {..}, ...}`` entries from truncated output.

    This parser is intentionally conservative:
    - Only salvages entries whose value is a balanced JSON object ``{...}``.
    - Stops at the first incomplete value object.
    - Ignores entries that cannot be ``json.loads``'d.
    """

    start = text.find("{")
    if start < 0:
        return None

    i = start + 1
    n = len(text)
    out: Dict[str, Any] = {}

    def skip_ws_and_commas(pos: int) -> int:
        while pos < n and text[pos] in " \t\r\n,":
            pos += 1
        return pos

    def parse_json_string(pos: int) -> Tuple[str | None, int]:
        if pos >= n or text[pos] != '"':
            return None, pos
        j = pos + 1
        escaped = False
        while j < n:
            ch = text[j]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                return text[pos : j + 1], j + 1
            j += 1
        return None, pos

    def extract_balanced_object(pos: int) -> Tuple[str | None, int]:
        if pos >= n or text[pos] != "{":
            return None, pos
        depth = 0
        j = pos
        in_str = False
        escaped = False
        while j < n:
            ch = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[pos : j + 1], j + 1
            j += 1
        return None, pos

    while True:
        i = skip_ws_and_commas(i)
        if i >= n:
            break
        if text[i] == "}":
            break

        key_raw, i2 = parse_json_string(i)
        if key_raw is None:
            break
        try:
            key = json.loads(key_raw)
        except Exception:
            break
        i = skip_ws_and_commas(i2)
        if i >= n or text[i] != ":":
            break
        i = skip_ws_and_commas(i + 1)

        val_raw, i3 = extract_balanced_object(i)
        if val_raw is None:
            break

        try:
            out[str(key)] = json.loads(val_raw)
        except Exception:
            pass
        i = i3

    return out or None


def extract_json_block(text: str) -> str | None:
    """Return the largest balanced {...} block, or None if not found.

    If the JSON is incomplete (truncated), attempts to repair it by:
    1) Finding the last complete object entry
    2) Adding missing closing braces/quotes
    """

    start = None
    depth = 0
    last_good = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0 and start is None:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last_good = i

    if start is not None and last_good is not None and last_good >= start:
        return text[start : last_good + 1]

    if start is not None:
        candidate = text[start:]

        best_pos = -1
        for pattern in ["},\n", '},"', "}, "]:
            idx = candidate.rfind(pattern)
            if idx > best_pos:
                best_pos = idx

        if best_pos > 0:
            repaired = candidate[: best_pos + 1] + "}"
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass

        last_comma_idx = candidate.rfind("},")
        if last_comma_idx > 0:
            repaired = candidate[: last_comma_idx + 1] + "}"
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass

        if "<im_end>" in text:
            open_braces = candidate.count("{")
            close_braces = candidate.count("}")
            missing = open_braces - close_braces
            if missing > 0:
                repaired = candidate + "}" * missing
                return repaired

    return None


def _load_json_loose(payload: str) -> Any | None:
    """Attempt to load JSON with a few lenient repairs."""

    try:
        return json.loads(payload)
    except Exception:
        pass

    fixed = re.sub(r",\s*([}\]])", r"\1", payload)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    return None


def parse_prediction(text: str) -> List[Dict[str, Any]]:
    """Parse model output JSON into a list of objects with integer coords."""

    obj = load_prediction_dict(text)
    if obj is None:
        return []

    parsed: List[Dict[str, Any]] = []
    for key, val in sorted(obj.items(), key=lambda kv: str(kv[0])):
        if not isinstance(val, dict):
            continue
        if "line" in val or "line_points" in val:
            parsed.append(
                {
                    "desc": str(val.get("desc", "")),
                    "line": val.get("line"),
                    "line_points": val.get("line_points"),
                }
            )
            continue

        geom_keys = [g for g in GEOM_KEYS if g in val]
        if len(geom_keys) != 1:
            continue
        gtype = geom_keys[0]

        pts_raw = flatten_points(val.get(gtype))
        if pts_raw is None or len(pts_raw) % 2 != 0:
            continue

        points, had_tokens = coerce_point_list(pts_raw)
        if points is None:
            continue

        # If coord tokens were present, enforce the 0..999 contract. For pure
        # numeric outputs (text-mode) allow pixel-space values so downstream
        # heuristics can decide how to scale.
        if had_tokens and any(v < 0 or v > MAX_BIN for v in points):
            continue

        ints = [int(round(v)) for v in points]
        parsed.append(
            {
                "desc": str(val.get("desc", "")),
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
    for i, v in enumerate(points):
        bound = w - 1.0 if i % 2 == 0 else h - 1.0
        out.append(min(max(float(v), 0.0), bound))
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
