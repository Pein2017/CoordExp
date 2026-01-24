#!/usr/bin/env python3
"""Visualize bbox/poly geometries for a single CoordExp JSONL record.

This is meant for quick sanity checks on crowded LVIS samples:
- Draw bbox instances (red)
- Draw poly instances (green)

Example:
  python public_data/scripts/visualize_jsonl_bbox_poly.py \\
    --jsonl public_data/lvis/rescale_32_768_poly_20/val.mix_hull_cap20.raw.jsonl \\
    --line 311 \\
    --out output/vis_lvis/008691_cap20.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw


def _load_line(path: Path, line_no: int) -> Dict[str, Any]:
    if line_no <= 0:
        raise ValueError("--line must be 1-based and >= 1")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i == line_no:
                return json.loads(line)
    raise FileNotFoundError(f"Line {line_no} not found in {path}")


def _pair_points(flat: List[int]) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for i in range(0, len(flat), 2):
        pts.append((int(flat[i]), int(flat[i + 1])))
    return pts


def _bbox_to_poly(b: List[int]) -> List[Tuple[int, int]]:
    if len(b) != 4:
        return []
    x1, y1, x2, y2 = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--line", type=int, required=True, help="1-based line number")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-objects", type=int, default=None, help="Optional cap for drawing only")
    ap.add_argument("--draw-bbox", type=int, default=1, help="0/1")
    ap.add_argument("--draw-poly", type=int, default=1, help="0/1")
    ap.add_argument("--width", type=int, default=2, help="Line width (pixels)")
    args = ap.parse_args()

    rec = _load_line(args.jsonl, int(args.line))
    images = rec.get("images") or []
    if not isinstance(images, list) or not images:
        raise ValueError("Record has no images[]")
    image_rel = str(images[0])
    img_path = (args.jsonl.parent / image_rel).resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    draw_bbox = bool(int(args.draw_bbox))
    draw_poly = bool(int(args.draw_poly))
    lw = int(args.width)

    bbox_color = (255, 64, 64, 180)
    poly_color = (64, 255, 64, 180)

    objects = rec.get("objects") or []
    if not isinstance(objects, list):
        objects = []
    if args.max_objects is not None:
        objects = objects[: int(args.max_objects)]

    n_bbox = 0
    n_poly = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if draw_bbox and "bbox_2d" in obj:
            b = obj.get("bbox_2d") or []
            if isinstance(b, list) and len(b) == 4:
                pts = _bbox_to_poly([int(x) for x in b])
                if pts:
                    draw.polygon(pts, outline=bbox_color, width=lw)
                    n_bbox += 1
        if draw_poly and "poly" in obj:
            p = obj.get("poly") or []
            if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0:
                pts = _pair_points([int(x) for x in p])
                if len(pts) >= 3:
                    draw.polygon(pts, outline=poly_color, width=lw)
                    n_poly += 1

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    vis = Image.alpha_composite(img, overlay).convert("RGB")
    vis.save(out)

    print(
        json.dumps(
            {
                "jsonl": str(args.jsonl),
                "line": int(args.line),
                "image": str(img_path),
                "size": list(vis.size),
                "drawn_bbox": n_bbox,
                "drawn_poly": n_poly,
                "objects_in_record": len(rec.get("objects") or []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

