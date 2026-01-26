"""
Shared parsing and coordinate utilities for CoordExp evaluation.

Factored from ``vis_tools/vis_coordexp.py`` so detection evaluation and
visualization can stay in sync.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from src.common.geometry import (
    MAX_BIN,
    coerce_point_list,
    flatten_points,
)

GEOM_KEYS = ("bbox_2d", "poly")


def extract_json_block(text: str) -> str | None:
    """Return the largest balanced {...} block, or None if not found.

    If the JSON is incomplete (truncated), attempts to repair it by:
    1. Finding the last complete object entry
    2. Adding missing closing braces/quotes
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

    # If no balanced block found, try to repair incomplete JSON
    if start is not None:
        candidate = text[start:]

        # Strategy 1: Find the last complete object entry
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

        # Strategy 2: Find any "}," pattern (more lenient)
        last_comma_idx = candidate.rfind("},")
        if last_comma_idx > 0:
            repaired = candidate[: last_comma_idx + 1] + "}"
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass

        # Strategy 3: Count braces and try to close (last resort)
        # If the generator emitted an explicit terminator like "<im_end>", we're
        # more tolerant and will attempt to close unmatched braces. However when
        # the output was truncated (no "<im_end>"), prefer dropping the final
        # incomplete object instead of fabricating closing braces that can leave
        # a malformed last entry. Only attempt brace-closing as a last resort
        # if the terminator exists.
        if "<im_end>" in text:
            open_braces = candidate.count("{")
            close_braces = candidate.count("}")
            missing = open_braces - close_braces
            if missing > 0:
                repaired = candidate + "}" * missing
                return repaired

    return None


def _load_json_loose(payload: str) -> Any | None:
    """Attempt to load JSON with a few lenient repairs (trailing commas, missing braces)."""

    try:
        return json.loads(payload)
    except Exception:
        pass

    # Remove trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", payload)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    return None


def parse_prediction(text: str) -> List[Dict[str, Any]]:
    """Parse model output JSON into a list of objects with integer coords."""
    block = extract_json_block(text)
    if not block:
        # Try closing braces heuristically
        # Only attempt to heuristically close unmatched braces when the model
        # emitted an explicit terminator (e.g. "<im_end>"). If the output was
        # truncated (no terminator), prefer returning no block so downstream code
        # can treat the prediction as empty / malformed rather than repairing a
        # likely-broken final object.
        if "<im_end>" in text and "{" in text:
            start = text.find("{")
            candidate = text[start:]
            missing = candidate.count("{") - candidate.count("}")
            if missing > 0:
                block = candidate + ("}" * missing)
        if not block:
            return []
    obj = _load_json_loose(block)
    if obj is None:
        return []

    if not isinstance(obj, dict):
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
        ints, had_tokens = coerce_point_list(pts_raw)
        if ints is None:
            continue
        # If coord tokens were present, enforce the 0..999 contract. For pure
        # numeric outputs (text-mode) allow pixel-space values so downstream
        # heuristics can decide how to scale.
        if had_tokens and any(v < 0 or v > MAX_BIN for v in ints):
            continue
        ints = [int(round(v)) for v in ints]
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
        # Small images (<=999px) with numeric coords are most likely pixel-space
        return True
    return False


def clamp_points(points: Sequence[float], width: float, height: float) -> List[float]:
    """Clamp coordinates to image bounds."""
    out: List[float] = []
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    for i, v in enumerate(points):
        bound = w - 1.0 if i % 2 == 0 else h - 1.0
        out.append(min(max(float(v), 0.0), bound))
    return out


def ints_to_pixels(ints: Sequence[int], width: float, height: float) -> List[float]:
    """Convert normalized 0-999 coords to pixel coordinates."""
    out: List[float] = []
    denom_x = max(1.0, float(width) - 1.0)
    denom_y = max(1.0, float(height) - 1.0)
    for i, v in enumerate(ints):
        frac = float(v) / float(MAX_BIN)
        if i % 2 == 0:
            out.append(frac * denom_x)
        else:
            out.append(frac * denom_y)
    return out


def pair_points(points: Sequence[float]) -> List[Tuple[float, float]]:
    assert len(points) % 2 == 0
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
