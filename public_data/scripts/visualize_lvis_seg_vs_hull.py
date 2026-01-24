#!/usr/bin/env python3
"""Visualize LVIS raw multi-part segmentation vs convex-hull enclosure.

We render side-by-side images:
  - Left: raw LVIS segmentation (all parts) (blue)
  - Right: convex hull of all segmentation vertices (red)

This helps sanity-check how much detail is lost when approximating complex,
possibly multi-part objects with a single outer enclosure polygon.

Example:
  PYTHONPATH=. conda run -n ms python public_data/scripts/visualize_lvis_seg_vs_hull.py \\
    --lvis-json public_data/lvis/raw/annotations/lvis_v1_val.json \\
    --images-dir public_data/lvis/rescale_32_768_poly_20/images \\
    --out-dir output/vis_lvis/raw_vs_hull \\
    --num 8 --min-parts 2 --min-area 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from src.datasets.preprocessors.resize import smart_resize


Point = Tuple[float, float]


def _extract_coco_file_name(image: Dict[str, Any]) -> Optional[str]:
    file_name = image.get("coco_file_name") or image.get("file_name")
    if isinstance(file_name, str) and file_name:
        return file_name
    coco_url = image.get("coco_url")
    if isinstance(coco_url, str) and coco_url:
        parts = coco_url.split("/")
        if len(parts) >= 2:
            return "/".join(parts[-2:])
    return None


def _pair_flat(coords: Sequence[float]) -> List[Point]:
    pts: List[Point] = []
    for i in range(0, len(coords), 2):
        pts.append((float(coords[i]), float(coords[i + 1])))
    # Strip duplicate closing point if present.
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def _poly_area(poly: Sequence[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _convex_hull(points: Sequence[Point]) -> List[Point]:
    """Monotonic chain convex hull. Returns CCW hull without duplicate last point."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return list(pts)

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _scale_points(points: Sequence[Point], sx: float, sy: float) -> List[Point]:
    return [(p[0] * sx, p[1] * sy) for p in points]


def _clamp_points(points: Sequence[Point], width: int, height: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for x, y in points:
        xi = max(0, min(width - 1, int(round(x))))
        yi = max(0, min(height - 1, int(round(y))))
        out.append((xi, yi))
    return out


def _draw_poly_outline(draw: ImageDraw.ImageDraw, pts: List[Tuple[int, int]], *, color, width: int) -> None:
    if len(pts) < 2:
        return
    # Close the loop.
    draw.line(pts + [pts[0]], fill=color, width=width)


@dataclass(frozen=True)
class Candidate:
    ann_id: int
    image_id: int
    cat: str
    image_path: Path
    orig_w: int
    orig_h: int
    tgt_w: int
    tgt_h: int
    n_parts: int
    total_vertices: int
    hull_vertices: int
    area_gt: float
    area_hull: float
    iou_hull: float


def _safe_cat_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip().lower())
    return cleaned[:64] if cleaned else "object"


def _load_json_bytes(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        import orjson

        return orjson.loads(raw)
    except Exception:
        return json.loads(raw.decode("utf-8"))


def _iter_valid_parts(segmentation: Any) -> Iterable[List[Point]]:
    if not isinstance(segmentation, list) or not segmentation:
        return
    for part in segmentation:
        if not isinstance(part, list) or len(part) < 6 or len(part) % 2 != 0:
            continue
        try:
            coords = [float(v) for v in part]
        except Exception:
            continue
        pts = _pair_flat(coords)
        if len(pts) >= 3:
            yield pts


def _collect_candidates(
    *,
    lvis_json: Path,
    images_dir: Path,
    min_parts: int,
    min_area: float,
    max_hull_verts: Optional[int],
) -> List[Candidate]:
    data = _load_json_bytes(lvis_json)
    categories = data.get("categories") or []
    images = data.get("images") or []
    annotations = data.get("annotations") or []

    cat_id_to_name: Dict[int, str] = {}
    for c in categories:
        try:
            cat_id_to_name[int(c["id"])] = str(c["name"])
        except Exception:
            continue

    image_id_to_info: Dict[int, Dict[str, Any]] = {}
    for img in images:
        try:
            img_id = int(img["id"])
        except Exception:
            continue
        file_name = _extract_coco_file_name(img)
        if not file_name:
            continue
        image_id_to_info[img_id] = {
            "file_name": file_name,
            "width": int(img["width"]),
            "height": int(img["height"]),
        }

    out: List[Candidate] = []
    for ann in annotations:
        try:
            img_id = int(ann["image_id"])
            ann_id = int(ann["id"])
            cat_name = cat_id_to_name[int(ann["category_id"])]
        except Exception:
            continue

        info = image_id_to_info.get(img_id)
        if not info:
            continue

        area_gt = float(ann.get("area") or 0.0)
        if area_gt < float(min_area):
            continue

        parts = list(_iter_valid_parts(ann.get("segmentation")))
        if len(parts) < int(min_parts):
            continue

        pts_all: List[Point] = []
        total_vertices = 0
        for pts in parts:
            pts_all.extend(pts)
            total_vertices += len(pts)
        if len(pts_all) < 3:
            continue

        hull = _convex_hull(pts_all)
        if len(hull) < 3:
            continue
        if max_hull_verts is not None and len(hull) > int(max_hull_verts):
            continue

        area_hull = _poly_area(hull)
        if area_hull <= 0.0:
            continue

        iou_hull = area_gt / area_hull
        if iou_hull > 1.0:
            iou_hull = 1.0

        orig_w = int(info["width"])
        orig_h = int(info["height"])
        tgt_h, tgt_w = smart_resize(height=orig_h, width=orig_w)

        basename = os.path.basename(str(info["file_name"]))
        img_path = (images_dir / basename).resolve()
        if not img_path.exists():
            continue

        out.append(
            Candidate(
                ann_id=ann_id,
                image_id=img_id,
                cat=cat_name,
                image_path=img_path,
                orig_w=orig_w,
                orig_h=orig_h,
                tgt_w=int(tgt_w),
                tgt_h=int(tgt_h),
                n_parts=len(parts),
                total_vertices=int(total_vertices),
                hull_vertices=int(len(hull)),
                area_gt=float(area_gt),
                area_hull=float(area_hull),
                iou_hull=float(iou_hull),
            )
        )

    return out


def _render_candidate(
    cand: Candidate,
    *,
    lvis_json: Path,
    images_dir: Path,
    out_dir: Path,
    line_width: int,
    pad: int,
) -> Path:
    data = _load_json_bytes(lvis_json)
    anns = data.get("annotations") or []

    ann = None
    for a in anns:
        try:
            if int(a.get("id")) == int(cand.ann_id):
                ann = a
                break
        except Exception:
            continue
    if ann is None:
        raise RuntimeError(f"annotation id not found: {cand.ann_id}")

    parts = list(_iter_valid_parts(ann.get("segmentation")))
    pts_all: List[Point] = []
    for pts in parts:
        pts_all.extend(pts)
    hull = _convex_hull(pts_all)

    # Scale original coords to resized coords (match images_dir images).
    sx = float(cand.tgt_w) / float(cand.orig_w)
    sy = float(cand.tgt_h) / float(cand.orig_h)
    parts_scaled = [_clamp_points(_scale_points(pts, sx, sy), cand.tgt_w, cand.tgt_h) for pts in parts]
    hull_scaled = _clamp_points(_scale_points(hull, sx, sy), cand.tgt_w, cand.tgt_h)

    img = Image.open(cand.image_path).convert("RGB")
    if img.size != (cand.tgt_w, cand.tgt_h):
        # Be robust to any unexpected mismatch.
        cand_w, cand_h = img.size
        cand = Candidate(**{**cand.__dict__, "tgt_w": cand_w, "tgt_h": cand_h})  # type: ignore[arg-type]

    left = img.copy().convert("RGBA")
    right = img.copy().convert("RGBA")

    # Semi-transparent overlays.
    overlay_left = Image.new("RGBA", left.size, (0, 0, 0, 0))
    overlay_right = Image.new("RGBA", right.size, (0, 0, 0, 0))
    dl = ImageDraw.Draw(overlay_left)
    dr = ImageDraw.Draw(overlay_right)

    seg_color = (64, 128, 255, 180)
    hull_color = (255, 64, 64, 200)

    for pts in parts_scaled:
        if len(pts) >= 3:
            _draw_poly_outline(dl, pts, color=seg_color, width=int(line_width))
    if len(hull_scaled) >= 3:
        _draw_poly_outline(dr, hull_scaled, color=hull_color, width=int(line_width + 1))

    left = Image.alpha_composite(left, overlay_left).convert("RGB")
    right = Image.alpha_composite(right, overlay_right).convert("RGB")

    # Compose side-by-side.
    w, h = left.size
    canvas = Image.new("RGB", (w * 2 + pad, h + pad * 2), (0, 0, 0))
    canvas.paste(left, (0, pad))
    canvas.paste(right, (w + pad, pad))

    draw = ImageDraw.Draw(canvas)
    title_left = "LVIS segmentation (multi-part)"
    title_right = "Convex hull enclosure"
    meta = "\n".join(
        [
            (
                f"cat={cand.cat} ann_id={cand.ann_id} image_id={cand.image_id} "
                f"parts={cand.n_parts} verts={cand.total_vertices} hull_verts={cand.hull_vertices} "
                f"iou_hull~{cand.iou_hull:.3f} (area_gt/area_hull)"
            ),
            f"orig={cand.orig_w}x{cand.orig_h} resized={cand.tgt_w}x{cand.tgt_h}",
        ]
    )
    draw.text((4, 2), title_left, fill=(255, 255, 255))
    draw.text((w + pad + 4, 2), title_right, fill=(255, 255, 255))
    draw.text((4, h + pad + 4), meta, fill=(255, 255, 255))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ann{cand.ann_id}_img{cand.image_id}_{_safe_cat_name(cand.cat)}.png"
    canvas.save(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize LVIS raw segmentation vs convex hull")
    ap.add_argument("--lvis-json", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num", type=int, default=8, help="How many examples to render")
    ap.add_argument("--min-parts", type=int, default=2, help="Require at least N segmentation parts")
    ap.add_argument("--min-area", type=float, default=1000.0, help="Min GT mask area (raw LVIS pixels)")
    ap.add_argument(
        "--max-hull-verts",
        type=int,
        default=None,
        help="Optional: only keep examples whose convex hull has <= this many vertices (e.g. 20)",
    )
    ap.add_argument("--line-width", type=int, default=3)
    ap.add_argument("--pad", type=int, default=8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cands = _collect_candidates(
        lvis_json=args.lvis_json,
        images_dir=args.images_dir,
        min_parts=int(args.min_parts),
        min_area=float(args.min_area),
        max_hull_verts=int(args.max_hull_verts) if args.max_hull_verts is not None else None,
    )
    if not cands:
        raise SystemExit("No candidates found; try lowering --min-parts or --min-area")

    # Pick a mix: worst (lowest IoU) + best (highest IoU) among multi-part.
    cands_sorted = sorted(cands, key=lambda c: c.iou_hull)
    k = int(args.num)
    worst = cands_sorted[: max(1, k // 2)]
    best = list(reversed(cands_sorted))[: max(1, k - len(worst))]
    picked = worst + best

    index: List[Dict[str, Any]] = []
    for cand in picked:
        out = _render_candidate(
            cand,
            lvis_json=args.lvis_json,
            images_dir=args.images_dir,
            out_dir=args.out_dir,
            line_width=int(args.line_width),
            pad=int(args.pad),
        )
        index.append(
            {
                "out": str(out),
                "image": str(cand.image_path),
                "cat": cand.cat,
                "ann_id": cand.ann_id,
                "image_id": cand.image_id,
                "parts": cand.n_parts,
                "total_vertices": cand.total_vertices,
                "hull_vertices": cand.hull_vertices,
                "iou_hull": cand.iou_hull,
                "orig_w": cand.orig_w,
                "orig_h": cand.orig_h,
                "tgt_w": cand.tgt_w,
                "tgt_h": cand.tgt_h,
            }
        )

    idx_path = args.out_dir / "index.json"
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(index)} visualizations to: {args.out_dir}")
    print(f"Index: {idx_path}")


if __name__ == "__main__":
    main()
