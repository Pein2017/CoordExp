"""
Sorting + canonicalization utilities for public_data converters.

These mirror the Qwen3-VL prompt spec used throughout CoordExp:

- Objects are ordered **top-to-bottom then left-to-right**.
- `bbox_2d` anchor: top-left corner (x1, y1).
- `poly` anchor: the *first* vertex (x1, y1) **after** polygon vertex canonicalization:
  - drop duplicated closing point if present
  - order vertices clockwise around centroid
  - rotate so the top-most (then left-most) vertex is first

LVIS conversion uses only `bbox_2d` and `poly` (no `line` support required),
but we keep `line` handling here for completeness and future-proofing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


def canonicalize_poly(points: Sequence[Any]) -> List[float]:
    """Return polygon points in canonical vertex order.

    Args:
        points: Flat list-like `[x1, y1, x2, y2, ...]` (numeric).

    Returns:
        Flat list of floats in canonical order.

    Raises:
        ValueError: if input is not an even-length list with >= 3 vertices.

    Notes:
        This intentionally does *not* preserve the original edge walk. It matches the
        Qwen3-VL conversion rule "sort vertices by angle around centroid" to make
        vertex order deterministic across converters that emit unordered corners.
    """
    if not isinstance(points, Sequence):
        raise ValueError(f"poly must be a sequence, got {type(points)!r}")

    pts_raw = list(points)
    if len(pts_raw) < 6 or len(pts_raw) % 2 != 0:
        raise ValueError(f"poly must have even length >= 6, got {len(pts_raw)}")

    verts: List[Tuple[float, float]] = []
    for i in range(0, len(pts_raw), 2):
        try:
            x = float(pts_raw[i])
            y = float(pts_raw[i + 1])
        except Exception as e:
            raise ValueError(f"poly contains non-numeric coord at idx {i}: {e}") from e
        verts.append((x, y))

    # Drop duplicate closing point (common in some sources).
    if len(verts) >= 2 and verts[0][0] == verts[-1][0] and verts[0][1] == verts[-1][1]:
        verts = verts[:-1]

    if len(verts) < 3:
        raise ValueError("poly must have >= 3 vertices after closing-point removal")

    cx = sum(x for x, _ in verts) / len(verts)
    cy = sum(y for _, y in verts) / len(verts)

    def angle_key(p: Tuple[float, float]) -> Tuple[float, float, float]:
        # NOTE: We intentionally follow Qwen3-VL's ordering rule for determinism.
        angle = math.atan2(p[1] - cy, p[0] - cx)
        normalized = (angle + 2 * math.pi) % (2 * math.pi)
        return (-normalized, p[1], p[0])

    ordered = sorted(verts, key=angle_key)

    # Rotate so the top-most then left-most vertex is first.
    top_left_idx = min(range(len(ordered)), key=lambda i: (ordered[i][1], ordered[i][0]))
    ordered = ordered[top_left_idx:] + ordered[:top_left_idx]

    flat: List[float] = []
    for x, y in ordered:
        flat.extend([x, y])
    return flat


def _first_xy(obj: Dict[str, Any]) -> Tuple[float, float]:
    """Geometry-specific anchor point for object ordering."""
    if obj.get("bbox_2d") is not None:
        bbox = obj["bbox_2d"]
        if isinstance(bbox, Sequence) and len(bbox) >= 2:
            try:
                return float(bbox[0]), float(bbox[1])
            except Exception:
                return (float("inf"), float("inf"))
        return (float("inf"), float("inf"))

    if obj.get("poly") is not None:
        poly = obj["poly"]
        if isinstance(poly, Sequence) and len(poly) >= 2:
            try:
                return float(poly[0]), float(poly[1])
            except Exception:
                return (float("inf"), float("inf"))
        return (float("inf"), float("inf"))

    if obj.get("line") is not None:
        # Not used for LVIS, but kept for prompt-spec completeness.
        coords = obj["line"]
        if not isinstance(coords, Sequence) or len(coords) < 2:
            return (float("inf"), float("inf"))
        try:
            pts = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]
        except Exception:
            return (float("inf"), float("inf"))
        leftmost = min(pts, key=lambda p: (p[0], p[1]))
        return leftmost[0], leftmost[1]

    return (float("inf"), float("inf"))


def sort_objects_tlbr(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort objects top-to-bottom then left-to-right (prompt spec)."""
    return sorted(list(objects), key=lambda o: (_first_xy(o)[1], _first_xy(o)[0]))

