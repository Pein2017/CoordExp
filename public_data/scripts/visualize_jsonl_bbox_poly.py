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
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


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


def _legend_lines(objects: List[Dict[str, Any]], *, mode: str) -> List[str]:
    if mode not in ("counts", "list"):
        raise ValueError(f"Unsupported legend mode: {mode}")

    if mode == "list":
        lines: List[str] = []
        for i, obj in enumerate(objects, start=1):
            desc = obj.get("desc")
            if isinstance(desc, str) and desc.strip():
                lines.append(f"{i:>3}: {desc.strip()}")
            else:
                lines.append(f"{i:>3}: <missing desc>")
        return lines

    counts: Counter[str] = Counter()
    for obj in objects:
        desc = obj.get("desc")
        if isinstance(desc, str):
            d = desc.strip()
            if d:
                counts[d] += 1

    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = []
    for desc, n in items:
        if n == 1:
            lines.append(desc)
        else:
            lines.append(f"{desc} x{n}")
    return lines


def _render_legend_panel(
    *,
    height: int,
    width: int,
    header_lines: List[str],
    body_lines: List[str],
    max_lines: int,
) -> Image.Image:
    panel = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    pad = 8
    line_h = int(getattr(font, "size", 10)) + 4
    y = pad

    def draw_line(text: str, *, bold: bool = False) -> None:
        nonlocal y
        if y + line_h > height - pad:
            return
        # Pillow default font has no bold; use a darker fill for headers.
        fill = (0, 0, 0) if not bold else (0, 0, 0)
        draw.text((pad, y), text, fill=fill, font=font)
        y += line_h

    for h in header_lines:
        draw_line(h, bold=True)

    if header_lines:
        y += 2  # small gap

    shown = 0
    for line in body_lines:
        if shown >= int(max_lines):
            break
        if y + line_h > height - pad:
            break
        draw_line(line)
        shown += 1

    remaining = max(0, len(body_lines) - shown)
    if remaining > 0 and y + line_h <= height - pad:
        draw_line(f"... (+{remaining} more)")

    return panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--line", type=int, required=True, help="1-based line number")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-objects", type=int, default=None, help="Optional cap for drawing only")
    ap.add_argument("--draw-bbox", type=int, default=1, help="0/1")
    ap.add_argument("--draw-poly", type=int, default=1, help="0/1")
    ap.add_argument("--width", type=int, default=2, help="Line width (pixels)")
    ap.add_argument(
        "--legend",
        action="store_true",
        help="Append a right-side legend panel listing object desc strings",
    )
    ap.add_argument(
        "--legend-mode",
        choices=["counts", "list"],
        default="counts",
        help="Legend content: 'counts' groups desc with xN; 'list' shows one line per object",
    )
    ap.add_argument(
        "--legend-width",
        type=int,
        default=320,
        help="Legend panel width (pixels) when --legend is set",
    )
    ap.add_argument(
        "--legend-max-lines",
        type=int,
        default=60,
        help="Max legend body lines to render",
    )
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

    if bool(args.legend):
        header = [
            f"line: {int(args.line)}",
            f"objects drawn: {len(objects)}",
        ]
        body = _legend_lines(objects, mode=str(args.legend_mode))
        panel = _render_legend_panel(
            height=vis.height,
            width=int(args.legend_width),
            header_lines=header,
            body_lines=body,
            max_lines=int(args.legend_max_lines),
        )
        combined = Image.new("RGB", (vis.width + panel.width, vis.height), (255, 255, 255))
        combined.paste(vis, (0, 0))
        combined.paste(panel, (vis.width, 0))
        vis = combined
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
